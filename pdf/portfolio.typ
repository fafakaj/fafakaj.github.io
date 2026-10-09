// Полное портфолио: обзор + каждый проект кейсом (задача, решение, результат, скриншоты).
#import "common.typ": *

#show: setup.with(title: "портфолио")

#header()
#about()
#education()
#achievements-section()
#skills()

#let paragraphs(s) = s.trim().split("\n\n").map(t => par(t.trim())).join()

#if projects.len() > 0 {
  for (i, p) in projects.enumerate() {
    if i == 0 { pagebreak(weak: true); [= Проекты] } else { fade-rule() }
    block(breakable: false, width: 100%, {
      block(below: 0.7em, sticky: true, {
        text(size: 17pt, link(site + "projects/" + p.slug + "/", text(fill: ink, p.title)))
        linebreak()
        event-line(p)
      })
      text(size: 10.5pt, p.summary)
      v(0.2em)
      stack-line(p)
      if p.at("links", default: none) != none { linebreak(); links-line(p) }
      for (key, heading) in (("task", "Задача"), ("solution", "Решение"), ("result", "Результат")) {
        let body = p.at(key, default: none)
        if body != none [
          == #heading
          #paragraphs(body)
        ]
      }
      let images = p.at("images", default: none)
      if images != none and images.len() > 0 [
        == Скриншоты
        #for img in images {
          figure(
            box(stroke: 0.5pt + line-col, radius: 3pt, clip: true, image("/" + img.src, width: 100%)),
            caption: text(size: 8.5pt, fill: muted, img.at("caption", default: "")),
            supplement: none,
            numbering: none,
          )
        }
      ]
    })
  }
}
