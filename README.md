# fafakaj.github.io

Портфолио Александра Лесницкого: сайт https://fafakaj.github.io и два PDF (резюме и полное портфолио).
Стиль сайта — по образцу simon.lecoq.io: вкладки Feed, Code, Resume, Social.

## Как обновить

1. Измените `data/portfolio.yaml`. Это единственный источник данных для сайта и PDF.
2. Скриншоты положите в `static/img/` и укажите в проекте: `src: static/img/<файл>.jpg`.
3. Сделайте push в `main`. GitHub Actions соберёт сайт и PDF и опубликует их.

## Сборка локально

```bash
pip install -r requirements.txt   # Jinja2, PyYAML, Pygments
python build.py                   # сайт и PDF в dist/ (нужен typst в PATH)
python build.py --no-pdf          # только сайт
python -m http.server -d dist     # просмотр: http://localhost:8000
```

## Устройство

| Путь | Что это |
|---|---|
| `data/portfolio.yaml` | данные: профиль, лента (Feed), проекты (Code), достижения, навыки, контакты |
| `templates/` | HTML-шаблоны Jinja2: `feed.html`, `code.html` (обзор и страница проекта), `resume.html`, `social.html`, `404.html` |
| `static/` | CSS, JS (почта для спам-ботов скрыта ROT13), иконки, картинки, фон и узор разделителя |
| `pdf/resume.typ` | резюме на одну страницу: слева проекты (`resume: true`), справа образование и навыки |
| `pdf/portfolio.typ` | полное портфолио: каждый проект — кейс со скриншотами |
| `pdf/fonts/` | шрифты Golos Text и JetBrains Mono (SIL OFL 1.1) для PDF |
| `tests/fixture.yaml` | тестовые данные: проект без скриншота и без кода, профиль без фото |
| `build.py` | сборка: YAML → `dist/` |

Тестовая сборка: `python build.py --data tests/fixture.yaml --out /tmp/dist-test`.
