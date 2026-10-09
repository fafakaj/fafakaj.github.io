// Подсветка пункта меню для раздела, который сейчас на экране (только на главной).
const links = [...document.querySelectorAll('#menu a[href*="#"]')];
const sections = links
  .map((a) => document.getElementById(a.hash.slice(1)))
  .filter(Boolean);

if (sections.length && "IntersectionObserver" in window) {
  const setActive = (id) => {
    for (const a of links) {
      const on = a.hash === `#${id}`;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    }
  };
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries.filter((e) => e.isIntersecting);
      if (visible.length) setActive(visible[0].target.id);
    },
    { rootMargin: "-20% 0px -70% 0px" },
  );
  sections.forEach((s) => observer.observe(s));
}
