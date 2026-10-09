"""Сборка портфолио: data/portfolio.yaml → сайт в dist/ и два PDF в dist/pdf/.

Запуск: python build.py [--data data/portfolio.yaml] [--out dist] [--no-pdf]
PDF собирает Typst (команда typst должна быть в PATH).
"""

import argparse
import datetime as dt
import shutil
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent
PDF = {"resume": "Lesnitsky_Resume.pdf", "portfolio": "Lesnitsky_Portfolio.pdf"}


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.setdefault("projects", [])
    data.setdefault("achievements", [])
    slugs = [p["slug"] for p in data["projects"]]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("Повтор slug в projects: у каждого проекта свой slug")
    return data


def menu(data: dict) -> list[dict]:
    items = [("about", "Обо мне"), ("skills", "Навыки")]
    if data["projects"]:
        items.append(("projects", "Проекты"))
    if data["achievements"]:
        items.append(("achievements", "Достижения"))
    items += [("contacts", "Контакты"), ("pdf", "PDF")]
    return [{"id": i, "label": label, "href": f"#{i}"} for i, label in items]


def build_site(data: dict, out: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    common = {
        **data,
        "menu": menu(data),
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

    render("index.html", "index.html", active=None)
    render("404.html", "404.html", active=None)
    for p in data["projects"]:
        render("project.html", f"projects/{p['slug']}/index.html", p=p, active="projects")

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
