// Резюме на одну страницу. Проекты: те, у кого resume: true (если таких нет — все).
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
    block(below: 0.85em, breakable: false, {
      link(site + "projects/" + p.slug + "/", p.title)
      [ — #p.summary]
      linebreak()
      event-line(p)
      linebreak()
      stack-line(p)
      if p.at("links", default: none) != none { h(0.6em); links-line(p) }
    })
  }
]

#achievements-section(extra-only: true, title: "Другие достижения")

#v(1fr)
#align(center, text(size: 9pt, fill: muted)[
  Подробно о каждом проекте: #link(site, site-short)
])
