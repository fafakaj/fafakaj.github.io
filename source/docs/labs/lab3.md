# Отчет по лабораторной работе №3
## Тема: CI/CD для статического сайта в SourceCraft

**Ссылки на репозиторий:** [GitHub](https://github.com/fafakaj/fafakaj.github.io) / [Sourcecraft](https://sourcecraft.dev/fafakaj/portfolio)

---

## 1. Цель работы
Настроить CI/CD-пайплайн для автоматической сборки статического сайта на базе MkDocs и его публикации в отдельную ветку release, используя:
- GitHub Actions — для сборки и деплоя на GitHub Pages
- SourceCraft — как альтернативную платформу для оркестрации задач сборки

---

## 2. Реализация
### 2.1. Структура проекта
```
fafakaj.github.io/
├── .github/workflows/ci.yml    # GitHub Actions workflow
├── .sourcecraft/
│    ├── ci.yaml
│    └── sites.yaml             # SourceCraft workflow
├── source/
│   ├── mkdocs.yml              # Конфигурация MkDocs
│   ├── docs/                   # Исходные файлы документации
│   └── ...                     
├── requirements.txt            # Зависимости Python
└── README.md
```

---

### 2.2. GitHub Actions Workflow 

```yml
name: Build and Deploy MkDocs

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build MkDocs
        run: |
          cd source
          mkdocs build --site-dir /tmp/mkdocs_build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: /tmp/mkdocs_build
          publish_branch: release
          force_orphan: true
```

---

### 2.3. SourceCraft Workflow

- `ci.yaml`
```yaml
on:
  push:
    workflows: build-site
    filter:
      branches: main

workflows:
  build-site:
    tasks:
      - name: build-and-publish-site
        cubes:
          - name: build-mkdocs-site
            image: docker.io/library/python:3.13-slim
            script:
              - python -m pip install --upgrade pip
              - "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi"
              - cd source
              - mkdocs build -d ../docs
              - echo "Сайт собран в папке /docs"
              - ls -la ../docs

          - name: publish-to-release-branch
            script:
              - git checkout -b release
              - git add .
              - "git commit -m \"feat: автоматическое обновление сайта\""
              - "git push origin release -f"
```
---

- `sites.yaml`

```yaml
site:
  root: docs
  ref: release
```

---
## 3. Проверка настройки
```python
git remote -v                           
origin  https://github.com/fafakaj/fafakaj.github.io.git (fetch)  
origin  https://github.com/fafakaj/fafakaj.github.io.git (push)  
sourcecraft     https://fafakaj:token@git.sourcecraft.dev/fafakaj/portfolio.git (fetch)  
sourcecraft     https://fafakaj:token@git.sourcecraft.dev/fafakaj/portfolio.git (push)
```

---

## 4. Ссылки на сайт
- https://fafakaj.github.io/
- https://fafakaj.sourcecraft.site/portfolio/

---

## 5. Вывод 
1. Настроен полностью автоматизированный пайплайн сборки.
2. Реализовано разделение исходников и артефактов: ветка main содержит только код, release — только готовый сайт.
3. Получен опыт работы с двумя системами оркестрации (GitHub Actions + SourceCraft).