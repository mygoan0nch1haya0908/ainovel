document.querySelectorAll("[data-profile-selector]").forEach((selector) => {
  const form = selector.closest("form");
  const select = selector.querySelector("[data-profile-select]");
  const consent = selector.querySelector("[data-profile-consent]");
  const destination = selector.querySelector("[data-profile-destination]");
  const model = form.elements.model_name;
  const provider = form.elements.provider_name;
  let legacyModel = model.value;
  let wasBound = false;
  const update = (initial = false) => {
    const bound = Boolean(select.value);
    if (bound && !wasBound) legacyModel = model.value;
    const option = select.selectedOptions[0];
    consent.checked = false;
    consent.required = bound;
    model.readOnly = bound;
    provider.disabled = bound;
    if (bound) {
      if (!initial || !model.value) model.value = option.dataset.model;
      destination.textContent = option.dataset.unavailable === "yes"
        ? "原选择不可用；请明确重新选择配置，不会自动改用环境 Provider。"
        : `认证目标：${option.dataset.target} · 模型：${option.dataset.model}`;
    } else {
      if (wasBound) model.value = legacyModel;
      destination.textContent = "使用环境配置 / 本地演示；请选择并核对下方模型。";
    }
    wasBound = bound;
  };
  select.addEventListener("change", () => update());
  update(true);
});
