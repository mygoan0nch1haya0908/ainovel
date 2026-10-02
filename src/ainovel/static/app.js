document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!window.confirm(form.dataset.confirm || "确认继续此操作？")) {
      event.preventDefault();
    }
  });
});

document.querySelectorAll("form[data-outline-setup]").forEach((form) => {
  const updateMode = () => {
    const mode = form.querySelector('input[name="setup_mode"]:checked')?.value;
    form.querySelectorAll("[data-outline-modes]").forEach((group) => {
      const active = group.dataset.outlineModes.split(" ").includes(mode);
      group.hidden = !active;
      group.querySelectorAll("input, textarea").forEach((field) => {
        field.disabled = !active;
      });
    });
    form.elements.book_outline.required = mode === "hierarchical";
    form.elements.stage_architecture.required = mode !== "single_chapter";
    form.elements.chapter_outline.required = mode === "single_chapter";
    form.querySelector("[data-chapter-outline-label]").textContent = mode === "single_chapter"
      ? "第一章提纲（独立单章必填）" : "单章大纲（可选，当前阶段第一章）";
  };
  form.querySelectorAll('input[name="setup_mode"]').forEach((radio) => {
    radio.addEventListener("change", updateMode);
  });
  updateMode();
  form.removeAttribute("novalidate");
});
