// Полное портфолио: обзор + каждый проект кейсом (задача, решение, результат, скриншоты).
#import "common.typ": *

#show: setup.with(title: "портфолио")

#header()
#about()
#education()
#skills()
#achievements-section()

#let paragraphs(s) = s.trim().split("\n\n").map(t => par(t.trim())).join()

#if projects.len() > 0 {
  for (i, p) in projects.enumerate() {
    pagebreak(weak: true)
    if i == 0 [= Проекты]
    block(below: 0.8em, {
      text(size: 9pt, fill: muted, kind-line(p))
      linebreak()
      text(size: 16pt, weight: "bold", link(site + "projects/" + p.slug + "/", text(fill: ink, p.title)))
    })
    text(size: 10.5pt, p.summary)
    v(0.3em)
    if p.at("stack", default: none) != none [#label-line("Стек", p.stack.join(", "))]
    links-line(p)
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
          box(stroke: 0.5pt + rule, image("/" + img.src, width: 100%)),
          caption: img.at("caption", default: none),
          supplement: none,
          numbering: none,
        )
      }
    ]
  }
}
