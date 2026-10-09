// Общие стили и блоки для резюме и полного портфолио.
// Палитра — светлая тема сайта (образец cmu-agents.com).

#let accent = rgb("#8b0000")
#let ink = rgb("#333333")
#let muted = rgb("#666666")
#let rule = rgb("#e0e0e0")

#let data = yaml(sys.inputs.at("data", default: "/data/portfolio.yaml"))
#let person = data.person
#let site = data.site_url
#let projects = data.at("projects", default: none)
#let projects = if projects == none { () } else { projects }
#let achievements = data.at("achievements", default: none)
#let achievements = if achievements == none { () } else { achievements }

#let setup(title: "", doc) = {
  set document(title: person.full_name + " — " + title, author: person.full_name)
  set page(
    paper: "a4",
    margin: (x: 1.8cm, top: 1.6cm, bottom: 1.8cm),
    footer: context {
      set text(size: 8pt, fill: muted)
      person.full_name + " — " + title
      h(1fr)
      counter(page).display("1 из 1", both: true)
    },
  )
  set text(font: "Roboto", size: 10pt, fill: ink, lang: "ru")
  set par(leading: 0.62em, spacing: 0.9em)
  show link: set text(fill: accent)
  show heading.where(level: 1): it => block(
    width: 100%,
    above: 1.4em,
    below: 0.8em,
    stroke: (bottom: 1.2pt + accent),
    inset: (bottom: 0.35em),
    text(size: 13pt, weight: "medium", it.body),
  )
  show heading.where(level: 2): it => block(
    above: 1.1em,
    below: 0.55em,
    text(size: 11pt, weight: "bold", it.body),
  )
  doc
}

// Шапка: имя акцентом, роль, контакты одной строкой.
#let header() = {
  text(size: 22pt, weight: "bold", fill: accent, person.full_name)
  v(0.15em)
  let edu = data.education.at(0)
  text(size: 11pt, fill: muted)[#person.role \@ #edu.university]
  v(0.3em)
  set text(size: 9.5pt)
  data.contacts.map(c => [#c.label: #link(c.url, c.value)]).join(h(0.6em) + text(fill: muted, "·") + h(0.6em))
  [ #h(0.6em) #text(fill: muted, "·") #h(0.6em) Сайт: #link(site, site.trim("https://", at: start).trim("/", at: end))]
}

// Пункт списка: серая дата, жирный заголовок, текст.
#let entry(date: none, title: none, body) = block(
  breakable: false,
  below: 1.3em,
  {
    if date != none { text(size: 9pt, fill: muted, date); linebreak() }
    if title != none { text(size: 11pt, weight: "bold", title); v(0.1em) }
    body
  },
)

#let label-line(label, value) = [*#label:* #value \ ]

#let kind-line(p) = {
  let s = str(p.year)
  if p.at("kind", default: none) != none { s += ", " + lower(p.kind) }
  s
}

#let about() = [
  = Обо мне
  #person.about
]

#let education() = [
  = Образование
  #for e in data.education {
    entry(date: e.period, title: [#e.university, #e.course])[
      #label-line("Факультет", e.faculty)
      #label-line("Направление", e.direction)
      #label-line("Программа", e.program)
      *Трек:* #e.track
    ]
  }
]

#let skills() = [
  = Навыки
  #grid(
    columns: (auto, 1fr),
    column-gutter: 1em,
    row-gutter: 0.6em,
    ..data.skills.map(s => (strong(s.group), s.items.join(", "))).flatten(),
  )
]

#let links-line(p) = {
  let links = p.at("links", default: none)
  if links != none and links.len() > 0 {
    [*Ссылки:* ]
    links.map(l => link(l.url, l.label)).join(text(fill: muted, " · "))
  }
}

#let achievements-section() = {
  if achievements.len() > 0 [
    = Хакатоны и достижения
    #for a in achievements {
      let title = if a.at("url", default: none) != none { link(a.url, a.title) } else { a.title }
      entry(date: a.date, title: title)[
        #if a.at("result", default: none) != none [#label-line("Результат", a.result)]
        #a.at("description", default: "")
      ]
    }
  ]
}
