"""Сборка портфолио: data/portfolio.yaml → сайт в dist/ и два PDF в dist/pdf/.

Запуск: python build.py [--data data/portfolio.yaml] [--out dist] [--no-pdf]
PDF собирает Typst (команда typst должна быть в PATH).
"""

import argparse
import codecs
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = Path(__file__).resolve().parent
PDF = {"resume": "Lesnitsky_Resume.pdf", "portfolio": "Lesnitsky_Portfolio.pdf"}
ICONS = ROOT / "static" / "icons" / "lucide"

# Вкладки сверху. Art и Memory из образца не нужны.
TABS = [
    {"id": "feed", "label": "Feed", "href": "", "icon": "cloud"},
    {"id": "code", "label": "Code", "href": "code/", "icon": "square-terminal"},
    {"id": "resume", "label": "Resume", "href": "resume/", "icon": "cpu"},
    {"id": "social", "label": "Social", "href": "social/", "icon": "share-2"},
]

MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
          "августа", "сентября", "октября", "ноября", "декабря"]
MONTHS_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
              "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+|mailto:[^)\s]+|/[^)\s]*)\)")


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("projects", "achievements", "feed"):
        data[key] = data.get(key) or []
    slugs = [p["slug"] for p in data["projects"]]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("Повтор slug в projects: у каждого проекта свой slug")
    for post in data["feed"]:
        if post.get("project") and post["project"] not in slugs:
            raise SystemExit(f"feed: нет проекта со slug {post['project']!r}")
    return data


def md(text: str) -> Markup:
    """Мини-разметка: абзацы через пустую строку, ссылки [текст](url)."""
    paras = []
    for para in str(text).strip().split("\n\n"):
        html = LINK.sub(lambda m: f'<a href="{m[2]}">{m[1]}</a>', str(escape(para.strip())))
        paras.append(f"<p>{html}</p>")
    return Markup("\n".join(paras))


def human_date(value) -> str:
    """2026-10-08 → «четверг, 8 октября 2026 г.»; «2025-10» → «октябрь 2025 г.»."""
    if isinstance(value, dt.date):
        return f"{WEEKDAYS[value.weekday()]}, {value.day} {MONTHS[value.month - 1]} {value.year} г."
    year, month = str(value).split("-")[:2]
    return f"{MONTHS_NOM[int(month) - 1]} {year} г."


def iso_date(value) -> str:
    return value.isoformat() if isinstance(value, dt.date) else str(value or "")


def icon(name: str) -> Markup:
    """SVG-иконка Lucide прямо в HTML: так она берёт цвет текста (currentColor)."""
    svg = (ICONS / f"{name}.svg").read_text(encoding="utf-8")
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    # Размеры и класс убираем только у самого <svg>: у <rect> внутри width/height нужны.
    head, rest = svg.split(">", 1)
    head = re.sub(r'\s(?:class|width|height)="[^"]*"', "", head)
    svg = head.replace("<svg", '<svg class="icon" aria-hidden="true" focusable="false"', 1) + ">" + rest
    return Markup(" ".join(svg.split()))


def code_html(snippet: dict) -> Markup:
    lexer = get_lexer_by_name(snippet.get("lang", "text"))
    return Markup(highlight(snippet["code"], lexer, HtmlFormatter(nowrap=True)))


def build_site(data: dict, out: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["md"] = md
    env.filters["rot13"] = lambda s: codecs.encode(str(s), "rot13")
    env.filters["human_date"] = human_date
    env.filters["iso_date"] = iso_date
    env.filters["code"] = code_html
    env.globals["icon"] = icon

    projects = data["projects"]
    by_date = sorted(projects, key=lambda p: iso_date(p.get("date")), reverse=True)
    common = {
        **data,
        "tabs": TABS,
        "pdf": PDF,
        "root": "/",
        "by_slug": {p["slug"]: p for p in projects},
        "featured": [p for p in by_date if p.get("featured")],
        "by_date": by_date,
        "updated": dt.date.today().strftime("%d.%m.%Y"),
    }

    def render(template: str, target: str, **extra) -> None:
        path = out / target
        path.parent.mkdir(parents=True, exist_ok=True)
        page = target.removesuffix("index.html")
        html = env.get_template(template).render(**common, path=page, **extra)
        path.write_text(html, encoding="utf-8")

    render("feed.html", "index.html", tab="feed")
    render("code.html", "code/index.html", tab="code", p=None)
    for p in projects:
        render("code.html", f"code/{p['slug']}/index.html", tab="code", p=p)
    render("resume.html", "resume/index.html", tab="resume")
    render("social.html", "social/index.html", tab="social")
    render("404.html", "404.html", tab=None)

    shutil.copytree(ROOT / "static", out / "static", dirs_exist_ok=True)
    (out / ".nojekyll").write_text("")


def build_pdf(data_path: Path, out: Path) -> None:
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    data_rel = "/" + data_path.resolve().relative_to(ROOT).as_posix()
    for name, filename in PDF.items():
        subprocess.run(
            [
                "typst", "compile",
                "--root", str(ROOT),
                "--font-path", str(ROOT / "pdf" / "fonts"),
                "--ignore-system-fonts",
                "--input", f"data={data_rel}",
                str(ROOT / "pdf" / f"{name}.typ"),
                str(out / "pdf" / filename),
            ],
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "portfolio.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--no-pdf", action="store_true", help="не собирать PDF")
    args = ap.parse_args()

    data = load(args.data)
    if args.out.exists():
        shutil.rmtree(args.out)
    build_site(data, args.out)
    if not args.no_pdf:
        build_pdf(args.data, args.out)
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
