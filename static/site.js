// Почта: в HTML лежит в ROT13, так её не видят спам-боты. Скрипт ставит настоящий mailto.
const rot13 = (s) =>
  s.replace(/[a-z]/gi, (c) => {
    const base = c <= "Z" ? 65 : 97;
    return String.fromCharCode(((c.charCodeAt(0) - base + 13) % 26) + base);
  });

for (const link of document.querySelectorAll("[data-email]")) {
  link.href = `mailto:${rot13(link.dataset.email)}`;
}

// Меню вкладки Code: на широком экране открыто всегда, на узком — свёрнуто до нажатия.
const aside = document.querySelector("details.aside");
if (aside) {
  const wide = window.matchMedia("(min-width: 1180px)");
  const sync = () => { aside.open = wide.matches; };
  sync();
  wide.addEventListener("change", sync);
}
