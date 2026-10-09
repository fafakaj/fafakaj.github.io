// Резюме на 1–2 страницы. Проекты: те, у кого resume: true (если таких нет — все).
#import "common.typ": *

#show: setup.with(title: "резюме")

#header()
#about()
#education()
#skills()

#let picked = projects.filter(p => p.at("resume", default: false) == true)
#let shown = if picked.len() > 0 { picked } else { projects }

#if shown.len() > 0 [
  = Проекты
  #for p in shown {
    entry(date: kind-line(p), title: link(site + "projects/" + p.slug + "/", p.title))[
      #p.summary \
      #if p.at("stack", default: none) != none [#label-line("Стек", p.stack.join(", "))]
      #links-line(p)
    ]
  }
]

#achievements-section()

#v(1fr)
#align(center, text(size: 9pt, fill: muted)[
  Подробное портфолио со всеми проектами: #link(site, site.trim("https://", at: start).trim("/", at: end))
])
