// Display-only controls: no storage, network calls or form submission.
document.querySelectorAll("[data-disclosure-scope]").forEach((scope) => {
  scope.querySelectorAll("[data-disclosure-action]").forEach((button) => {
    button.hidden = false;
    button.addEventListener("click", () => {
      scope.querySelectorAll("details[data-reading-panel]").forEach((panel) => {
        panel.open = button.dataset.disclosureAction === "expand";
      });
    });
  });
});
document.addEventListener("invalid", (event) => {
  let parent = event.target.parentElement;
  while (parent) {
    if (parent.matches("details")) parent.open = true;
    parent = parent.parentElement;
  }
}, true);
