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

ROOT = Path(__file__).resolve().parent
PDF = {"resume": "Lesnitsky_Resume.pdf", "portfolio": "Lesnitsky_Portfolio.pdf"}

# Вкладки сверху на каждой странице.
TABS = [
    {"id": "home", "label": "главная", "href": ""},
    {"id": "projects", "label": "проекты", "href": "projects/"},
    {"id": "social", "label": "контакты", "href": "social/"},
    {"id": "resume", "label": "резюме", "href": "resume/"},
]

# Файлы на вкладке «резюме»: превью — первая страница PDF в PNG.
DOCS = [
    {"id": "resume", "label": "резюме", "note": "PDF, 1 страница"},
    {"id": "portfolio", "label": "полное портфолио", "note": "PDF, все проекты со скриншотами"},
]


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.setdefault("projects", [])
    data.setdefault("achievements", [])
    slugs = [p["slug"] for p in data["projects"]]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("Повтор slug в projects: у каждого проекта свой slug")
    return data


LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+|mailto:[^)\s]+|/[^)\s]*)\)")


def md(text: str) -> Markup:
    """Мини-разметка для текста шкалы: абзацы через пустую строку, ссылки [текст](url)."""
    paras = []
    for para in str(text).strip().split("\n\n"):
        html = LINK.sub(lambda m: f'<a href="{m[2]}">{m[1]}</a>', str(escape(para.strip())))
        paras.append(f"<p>{html}</p>")
    return Markup("\n".join(paras))


def build_site(data: dict, out: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["md"] = md
    env.filters["rot13"] = lambda s: codecs.encode(str(s), "rot13")
    docs = []
    for d in DOCS:
        preview = f"{d['id']}-preview.png"
        docs.append({**d, "file": PDF[d["id"]],
                     "preview": preview if (out / "pdf" / preview).exists() else None})
    common = {
        **data,
        "tabs": TABS,
        "docs": docs,
        "pdf": PDF,
        "root": "/",
        "updated": dt.date.today().strftime("%d.%m.%Y"),
    }

    def render(template: str, target: str, **extra) -> None:
        path = out / target
        path.parent.mkdir(parents=True, exist_ok=True)
        page = target.removesuffix("index.html")
        html = env.get_template(template).render(**common, path=page, **extra)
        path.write_text(html, encoding="utf-8")

    render("index.html", "index.html", tab="home")
    render("projects.html", "projects/index.html", tab="projects")
    for p in data["projects"]:
        render("project.html", f"projects/{p['slug']}/index.html", tab="projects", p=p)
    render("social.html", "social/index.html", tab="social")
    render("resume.html", "resume/index.html", tab="resume")
    render("404.html", "404.html", tab=None)

    shutil.copytree(ROOT / "static", out / "static", dirs_exist_ok=True)
    (out / ".nojekyll").write_text("")


def build_pdf(data_path: Path, out: Path) -> None:
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    data_rel = "/" + data_path.resolve().relative_to(ROOT).as_posix()
    for name, filename in PDF.items():
        base = [
            "typst", "compile",
            "--root", str(ROOT),
            "--font-path", str(ROOT / "pdf" / "fonts"),
            "--ignore-system-fonts",
            "--input", f"data={data_rel}",
            str(ROOT / "pdf" / f"{name}.typ"),
        ]
        subprocess.run([*base, str(out / "pdf" / filename)], check=True)
        # Превью для вкладки «резюме»: первая страница в PNG.
        subprocess.run([*base, "--pages", "1", "--ppi", "110",
                        str(out / "pdf" / f"{name}-preview.png")], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "portfolio.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--no-pdf", action="store_true", help="не собирать PDF")
    args = ap.parse_args()

    data = load(args.data)
    if args.out.exists():
        shutil.rmtree(args.out)
    # PDF собираются первыми: сайт показывает их превью, если они есть.
    if not args.no_pdf:
        build_pdf(args.data, args.out)
    build_site(data, args.out)
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
