// Общие стили и блоки для резюме и полного портфолио.
// Стиль — как у сайта (образец simon.lecoq.io): тёмные плашки заголовков, бежево-серая палитра.
// Фон страницы белый, чтобы PDF было удобно печатать.

#let ink = rgb("#4B4A3F")
#let paper = rgb("#D7D2B8")
#let muted = rgb("#7d7a68")
#let faint = rgb("#a19c86")
#let line-col = rgb("#B3AE99")
#let link-col = rgb("#003A6B")
#let venue = rgb("#4B4A3F")

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

#let setup(title: "", doc) = {
  set document(title: person.full_name + " — " + title, author: person.full_name)
  set page(
    paper: "a4",
    margin: (x: 1.9cm, top: 1.4cm, bottom: 1.6cm),
    footer: context {
      set text(size: 8pt, fill: faint)
      person.full_name + " — " + title
      h(1fr)
      counter(page).display("1 из 1", both: true)
    },
  )
  set text(font: "Roboto", size: 10pt, fill: ink, lang: "ru")
  set par(leading: 0.55em, spacing: 0.85em)
  show link: set text(fill: link-col)
  // Заголовок раздела: тёмная плашка с квадратом, как шапка карточки на сайте.
  show heading.where(level: 1): it => block(
    above: 1.15em, below: 0.6em, width: 100%, fill: ink,
    inset: (x: 8pt, top: 4pt, bottom: 5pt), sticky: true,
    text(size: 11.5pt, weight: "regular", fill: paper)[■ #it.body])
  show heading.where(level: 2): it => block(above: 1em, below: 0.45em, sticky: true,
    text(size: 10.5pt, weight: "bold", fill: ink)[■ #it.body])
  doc
}

// Разделитель, как под вкладками сайта: толстая линия и ряд точек.
#let fade-rule() = block(above: 1em, below: 1em, width: 100%, {
  line(length: 100%, stroke: 1.6pt + ink)
  v(2.5pt)
  box(width: 100%, height: 4pt, clip: true,
    repeat(box(width: 14pt, align(left, circle(radius: 1.1pt, fill: ink)))))
})

// Шапка: имя, курсивная фраза, контакты.
#let header() = {
  let items = data.contacts.map(c => [#c.label: #link(c.url, c.value)])
  items.push([Сайт: #link(site, site-short)])
  align(center, stack(
    spacing: 9pt,
    text(size: 24pt, weight: "regular", fill: ink, person.full_name),
    text(size: 11.5pt, style: "italic", fill: muted, person.tagline),
    text(size: 9.5pt, items.join(h(0.5em) + text(fill: faint, "|") + h(0.5em))),
  ))
  fade-rule()
}

#let about() = {
  text(size: 10.5pt, person.about)
}

// Пункт шкалы: серый период слева, серая вертикальная линия, текст справа.
#let entry(period, body) = grid(
  columns: (2.6cm, 1fr),
  column-gutter: 0pt,
  align(right, pad(right: 7pt, text(size: 8.5pt, fill: faint, period))),
  block(stroke: (left: 1.5pt + line-col), inset: (left: 10pt, bottom: 9pt), width: 100%, body),
)

#let education() = [
  = Образование
  #for e in data.education {
    entry(e.period)[
      #text(size: 11pt)[#link(e.url, e.university), #e.course] \
      #e.faculty \
      Направление «#e.direction», программа «#e.program», \
      трек «#e.track»
    ]
  }
]

#let skills() = [
  = Навыки
  #grid(
    columns: (auto, 1fr),
    column-gutter: 1em,
    row-gutter: 0.55em,
    ..data.skills.map(s => (text(fill: muted, s.group + ":"), s.items.join(", "))).flatten(),
  )
]

#let event-line(p) = {
  let ev = p.at("event", default: none)
  if ev != none { text(size: 9pt, fill: venue, style: "italic", ev) }
}

#let stack-line(p) = {
  let st = p.at("stack", default: none)
  if st != none { text(size: 9pt, fill: muted, st.join(", ")) }
}

#let links-line(p) = {
  let links = p.at("links", default: none)
  if links != none and links.len() > 0 {
    text(size: 9pt, links.map(l => link(full-url(l.url), l.label)).join(", "))
  }
}

// Достижения: левая серая полоса, название, зелёный результат — как .pub на сайте.
// extra-only: только достижения без страницы проекта (для резюме, где результат уже под проектом).
#let achievements-section(extra-only: false, title: "Хакатоны и достижения") = {
  let shown = if extra-only {
    achievements.filter(a => not a.at("url", default: "").starts-with("/code/"))
  } else { achievements }
  if shown.len() > 0 [
    #heading(title)
    #for a in shown {
      let url = a.at("url", default: none)
      let title = if url != none { link(full-url(url), a.title) } else { a.title }
      block(
        stroke: (left: 3pt + line-col),
        inset: (left: 8pt, y: 2pt),
        below: 0.65em,
        breakable: false,
        {
          title
          if a.at("result", default: none) != none { h(0.4em); text(fill: venue, weight: "bold", a.result) }
          linebreak()
          text(size: 9pt)[#a.date#if a.at("description", default: none) != none [. #a.description]]
        },
      )
    }
  ]
}
