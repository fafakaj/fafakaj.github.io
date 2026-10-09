// Почта по клику, как на karpathy.ai: в HTML она лежит в ROT13, так её не видят спам-боты.
const rot13 = (s) =>
  s.replace(/[a-z]/gi, (c) => {
    const base = c <= "Z" ? 65 : 97;
    return String.fromCharCode(((c.charCodeAt(0) - base + 13) % 26) + base);
  });

const button = document.getElementById("iemail");
const out = document.getElementById("demail");
if (button && out) {
  button.addEventListener("click", () => {
    const shown = out.classList.toggle("shown");
    button.setAttribute("aria-expanded", String(shown));
    if (shown && !out.firstChild) {
      const email = rot13(button.dataset.email);
      const link = document.createElement("a");
      link.href = `mailto:${email}`;
      link.textContent = email;
      out.append(link);
    }
  });
}
