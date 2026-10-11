// Почта на вкладке «контакты»: в HTML она лежит в ROT13, так её не видят спам-боты.
const rot13 = (s) =>
  s.replace(/[a-z]/gi, (c) => {
    const base = c <= "Z" ? 65 : 97;
    return String.fromCharCode(((c.charCodeAt(0) - base + 13) % 26) + base);
  });

for (const link of document.querySelectorAll("a[data-email]")) {
  const email = rot13(link.dataset.email);
  link.href = `mailto:${email}`;
  link.textContent = email;
}
