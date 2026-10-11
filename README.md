# fafakaj.github.io

Портфолио Александра Лесницкого: сайт https://fafakaj.github.io и два PDF (резюме и полное портфолио).
Сайт и PDF на английском. Основа сайта — karpathy.ai: фото, имя и временная шкала. Поверх — светло-бежевый фон, шрифт Geist и кнопки-«таблетки».
Вкладки: Home (фото, имя, временная шкала хакатонов), Projects (проекты и стек технологий), Social (контакты), Resume (превью и скачивание PDF).

## Как обновить

1. Измените `data/portfolio.yaml`. Это единственный источник данных для сайта и PDF.
2. Скриншоты положите в `static/img/` и укажите в проекте: `src: static/img/<файл>.jpg`.
3. Сделайте push в `main`. GitHub Actions соберёт сайт и PDF и опубликует их.

## Сборка локально

```bash
pip install -r requirements.txt   # Jinja2, PyYAML
python build.py                   # сайт, PDF и их превью в dist/ (нужен typst в PATH)
python build.py --no-pdf          # только сайт
python -m http.server -d dist     # просмотр: http://localhost:8000
```

## Устройство

| Путь | Что это |
|---|---|
| `data/portfolio.yaml` | данные: имя, подпись и фото для главной, образование, стек технологий, проекты, хакатоны (временная шкала, с логотипами), контакты |
| `templates/` | HTML-шаблоны Jinja2: `base.html` (вкладки), `index.html`, `projects.html`, `project.html`, `social.html`, `resume.html`, `404.html` |
| `static/` | CSS, JS (почта скрыта от спам-ботов ROT13, кнопка копирования), иконки, картинки; источники логотипов — `static/img/SOURCES.txt` |
| `pdf/resume.typ` | резюме на одну страницу: слева проекты (`resume: true`), справа образование и навыки |
| `pdf/portfolio.typ` | полное портфолио: каждый проект — кейс со скриншотами |
| `pdf/fonts/` | шрифты Golos Text и JetBrains Mono (SIL OFL 1.1) для PDF |
| `tests/` | тестовые данные с проектами для проверки вёрстки |
| `build.py` | сборка: YAML → `dist/` |

Тестовая сборка: `python build.py --data tests/fixture.yaml --out /tmp/dist-test`.
