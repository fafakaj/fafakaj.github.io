// Резюме на одну страницу. Проекты: те, у кого resume: true (если таких нет — все).
// Слева проекты, справа узкая колонка: образование, навыки, хакатоны.
#import "common.typ": *

#show: setup.with(title: "резюме")

#header()
#about()
#fade-rule()

#let picked = projects.filter(p => p.at("resume", default: false) == true)
#let shown = if picked.len() > 0 { picked } else { projects }

// Итог проекта без фразы «Финал хакатона, N место.»: место уже в строке события.
#let result-text(p) = {
  let r = p.at("result", default: none)
  if r == none { return none }
  let m = r.match(regex("^Финал хакатона, \d+ место\.\s*"))
  let rest = if m != none { r.slice(m.end) } else { r }
  if rest.trim() != "" { rest.trim() }
}

#let project(p) = block(below: 1.05em, breakable: false, {
  link(site + "code/" + p.slug + "/", text(size: 10.4pt, weight: 600, fill: accent, p.title))
  linebreak()
  event-line(p)
  v(-1pt)
  block(above: 0.5em, below: 0.45em, p.summary)
  let r = result-text(p)
  if r != none { block(above: 0em, below: 0.45em)[*Результат.* #r] }
  stack-line(p)
  if p.at("links", default: none) != none { h(0.6em); links-line(p) }
})

#let main = {
  if shown.len() > 0 [
    #block(above: 0pt, below: 0.75em, sticky: true, {
      text(size: 12pt, weight: 700, fill: ink)[Проекты]
      v(-0.35em)
      dots(thickness: 1.3pt, gap: 3.6pt)
    })
    #for p in shown { project(p) }
  ]
}

#let side = {
  set text(size: 8.8pt)
  show heading.where(level: 1): set block(above: 1.4em)
  {
    show heading.where(level: 1): set block(above: 0pt)
    education()
  }
  skills()
  // Хакатоны с проектом уже есть слева; здесь — только остальные.
  achievements-section(extra-only: true, title: "Другие достижения")
}

#grid(
  columns: (1fr, 5.6cm),
  column-gutter: 0.75cm,
  main,
  side,
)

#v(1fr)
#dots()
#v(4pt)
#align(center, contacts-line())
