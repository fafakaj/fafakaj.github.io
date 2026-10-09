# fafakaj.github.io

Портфолио Александра Лесницкого: сайт https://fafakaj.github.io и два PDF (резюме и полное портфолио).

## Как обновить

1. Измените `data/portfolio.yaml`. Это единственный источник данных для сайта и PDF.
2. Скриншоты положите в `static/img/` и укажите в проекте: `src: static/img/<файл>.png`.
3. Сделайте push в `main`. GitHub Actions соберёт сайт и PDF и опубликует их.

## Сборка локально

```bash
pip install -r requirements.txt   # Jinja2, PyYAML
python build.py                   # сайт и PDF в dist/ (нужен typst в PATH)
python build.py --no-pdf          # только сайт
python -m http.server -d dist     # просмотр: http://localhost:8000
```

## Устройство

| Путь | Что это |
|---|---|
| `data/portfolio.yaml` | данные: о себе, образование, навыки, проекты, достижения, контакты |
| `templates/` | HTML-шаблоны Jinja2: главная, страница проекта, 404 |
| `static/` | CSS, JS, иконка, картинки |
| `pdf/resume.typ` | резюме на 1–2 страницы (проекты с `resume: true`) |
| `pdf/portfolio.typ` | полное портфолио: каждый проект — кейс со скриншотами |
| `pdf/fonts/` | шрифт Roboto (SIL OFL 1.1) для PDF |
| `tests/` | тестовые данные с проектами для проверки вёрстки |
| `build.py` | сборка: YAML → `dist/` |

Тестовая сборка: `python build.py --data tests/fixture.yaml --out /tmp/dist-test`.
