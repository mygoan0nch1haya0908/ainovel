(() => {
  const form = document.getElementById('extraction-step');
  const button = document.getElementById('extraction-run');
  if (!form || !button) return;
  let running = false;
  form.addEventListener('submit', event => { if (running) event.preventDefault(); });
  button.addEventListener('click', async () => {
    if (running) return;
    running = true;
    button.disabled = true;
    const progress = document.getElementById('extraction-progress');
    try {
      for (let count = 0; count < 8; count++) {
        progress.textContent = `正在串行处理第 ${count + 1} 次请求，请勿重复提交。`;
        const response = await fetch(form.action, {method: 'POST', body: new FormData(form), headers: {'Accept': 'application/json'}});
        if (!response.ok) throw new Error('请求未完成，请刷新核对状态与用量；不会自动重试。');
        const result = await response.json();
        form.elements.revision.value = String(result.revision);
        if (result.status !== 'READY') break;
      }
      window.location.reload();
    } catch (error) {
      progress.textContent = '整理已停止。请刷新核对状态与用量，不会自动重试。';
    }
  });
})();
