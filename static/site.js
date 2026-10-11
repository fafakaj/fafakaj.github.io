// Почта на вкладке Social: в HTML она лежит в ROT13, так её не видят спам-боты.
const rot13 = (s) =>
  s.replace(/[a-z]/gi, (c) => {
    const base = c <= "Z" ? 65 : 97;
    return String.fromCharCode(((c.charCodeAt(0) - base + 13) % 26) + base);
  });

for (const link of document.querySelectorAll("a[data-email]")) {
  const email = rot13(link.dataset.email);
  link.href = `mailto:${email}`;
  link.querySelector(".cvalue").textContent = email;
}

// Кнопка копирования: копирует адрес и на 2 секунды показывает галочку.
for (const button of document.querySelectorAll("button[data-copy-email]")) {
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(rot13(button.dataset.copyEmail));
    } catch {
      return;
    }
    button.classList.add("done");
    button.setAttribute("aria-label", "Email address copied");
    setTimeout(() => {
      button.classList.remove("done");
      button.setAttribute("aria-label", "Copy email address");
    }, 2000);
  });
}
