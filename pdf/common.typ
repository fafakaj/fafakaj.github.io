// Общие стили и блоки для резюме и полного портфолио.
// Стиль: белый лист, кобальтовый акцент, точечные разделители, навыки чипами.
// Шрифты: Golos Text (текст) и JetBrains Mono (стек технологий).

#let ink = rgb("#172033")
#let body-col = rgb("#2e3749")
#let muted = rgb("#6b7589")
#let cobalt = rgb("#2953d6")
#let chip-bg = rgb("#e7edfc")
#let dot-col = rgb("#9fb2ef")
#let tint = rgb("#f5f7fb")
// Старые имена: portfolio.typ рисует ими рамку скриншота.
#let line-col = dot-col
#let link-col = cobalt

#let data = yaml(sys.inputs.at("data", default: "/data/portfolio.yaml"))
#let person = data.person
#let site = data.site_url
#let site-short = site.trim("https://", at: start).trim("/", at: end)
#let list-or-empty(key) = {
  let v = data.at(key, default: none)
  if v == none { () } else { v }
}
#let projects = list-or-empty("projects")
#let achievements = list-or-empty("achievements")

// Ссылки сайта вида /code/x/ в PDF ведут на полный адрес.
#let full-url(url) = if url.starts-with("/") { site.trim("/", at: end) + url } else { url }

// Точечная линия — главный графический мотив.
#let dots(color: dot-col, thickness: 1.7pt, gap: 4.6pt) = line(
  length: 100%,
  stroke: (paint: color, thickness: thickness, cap: "round", dash: (0pt, gap)),
)

#let fade-rule() = block(above: 1.1em, below: 1.2em, dots())

#let setup(title: "", doc) = {
  set document(title: person.full_name + " — " + title, author: person.full_name)
  set page(
    paper: "a4",
    margin: (x: 1.6cm, top: 1.4cm, bottom: 1.5cm),
    footer: context {
      set text(size: 7.6pt, fill: muted)
      person.full_name + " — " + title
      h(1fr)
      counter(page).display("1 из 1", both: true)
    },
  )
  // Без переносов: при переносе Typst сопоставляет глиф дефиса с U+00AD,
  // и слова с дефисом («Веб-редактор») перестают находиться поиском и ATS.
  set text(font: "Golos Text", size: 9.6pt, fill: body-col, lang: "ru", hyphenate: false)
  set par(leading: 0.62em, spacing: 0.85em)
  show strong: set text(weight: 600, fill: ink)
  show link: set text(fill: cobalt)
  show raw: set text(font: "JetBrains Mono")
  // Заголовок раздела: жирный текст и короткая точечная линия под ним.
  show heading.where(level: 1): it => block(above: 1.3em, below: 0.75em, sticky: true, {
    text(size: 12pt, weight: 700, fill: ink, it.body)
    v(-0.35em)
    dots(thickness: 1.3pt, gap: 3.6pt)
  })
  show heading.where(level: 2): it => block(above: 1em, below: 0.45em, sticky: true,
    text(size: 10pt, weight: 600, fill: cobalt, it.body))
  doc
}

#let chip(body) = box(
  fill: chip-bg, inset: (x: 4.5pt), outset: (y: 3pt), radius: 2pt,
  text(size: 8.2pt, fill: ink, body),
)

#let chips(items) = {
  set par(leading: 1.05em)
  items.map(chip).join(h(3pt))
}

// Шапка: круглое фото, имя, роль, контакты текстом (без иконок — их не читает ATS).
#let header() = {
  let photo = person.at("photo", default: none)
  let contacts = data.contacts.map(c => [#text(fill: muted, c.label) #link(c.url, c.value)])
  contacts.push([#text(fill: muted)[Сайт] #link(site, site-short)])
  grid(
    columns: if photo != none { (auto, 1fr) } else { (1fr,) },
    column-gutter: 16pt,
    align: horizon,
    ..if photo != none {
      (box(width: 64pt, height: 64pt, radius: 32pt, clip: true, stroke: 1.5pt + chip-bg,
        image("/" + photo, width: 64pt, height: 64pt, fit: "cover")),)
    },
    {
      text(size: 22pt, weight: 700, fill: ink, person.name)
      v(1pt)
      text(size: 10.5pt, weight: 500, fill: cobalt, person.role)
      v(4pt)
      text(size: 8.8pt, contacts.join(h(12pt)))
    },
  )
  fade-rule()
}

#let about() = {
  text(size: 10pt, fill: ink, person.about)
}

// Пункт шкалы: период сверху, кобальтовая точка на вертикальной линии.
#let entry(period, body) = block(
  stroke: (left: 1.4pt + dot-col), inset: (left: 10pt, y: 1pt), below: 0.9em, breakable: false,
  {
    place(dx: -13.2pt, dy: 2.4pt, circle(radius: 2.6pt, fill: cobalt, stroke: 1pt + white))
    text(size: 8.4pt, weight: 600, fill: cobalt, period)
    linebreak()
    body
  },
)

#let education() = [
  = Образование
  #for e in data.education {
    entry(e.period)[
      #text(weight: 600)[#link(e.url, text(fill: ink, e.university)), #e.course] \
      #e.faculty \
      #text(size: 8.8pt, fill: muted)[Направление «#e.direction», программа «#e.program», трек «#e.track»]
    ]
  }
]

// Навыки: группа подписью, технологии чипами.
#let skills() = [
  = Навыки
  #for s in data.skills {
    block(below: 1em, breakable: false, {
      text(size: 8.4pt, weight: 600, fill: muted, s.group)
      v(1pt)
      chips(s.items)
    })
  }
]

#let event-line(p) = {
  let ev = p.at("event", default: none)
  if ev != none { text(size: 8.8pt, weight: 500, fill: ink, ev) }
}

#let stack-line(p) = {
  let st = p.at("stack", default: none)
  if st != none { text(size: 7.8pt, fill: muted, raw(st.join(", "))) }
}

#let links-line(p) = {
  let links = p.at("links", default: none)
  if links != none and links.len() > 0 {
    text(size: 8.8pt, links.map(l => link(full-url(l.url), l.label)).join(", "))
  }
}

// Достижения: название, результат кобальтом, дата и описание мелко.
// extra-only: только достижения без страницы проекта.
#let achievements-section(extra-only: false, title: "Хакатоны и достижения") = {
  let shown = if extra-only {
    achievements.filter(a => not a.at("url", default: "").starts-with("/code/"))
  } else { achievements }
  if shown.len() > 0 [
    #heading(title)
    #for a in shown {
      let url = a.at("url", default: none)
      let name = if url != none { link(full-url(url), text(fill: ink, a.title)) } else { a.title }
      block(below: 0.7em, breakable: false, {
        text(weight: 600, fill: ink, name)
        if a.at("result", default: none) != none { h(0.4em); text(fill: cobalt, weight: 600, a.result) }
        linebreak()
        text(size: 8.4pt, fill: muted)[#a.date#if a.at("description", default: none) != none [. #a.description]]
      })
    }
  ]
}
