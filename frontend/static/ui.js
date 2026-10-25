// Show/hide password buttons: <button class="eye" data-for="inputId">
document.addEventListener('click', e => {
  const b = e.target.closest('button.eye');
  if (!b) return;
  const input = document.getElementById(b.dataset.for);
  if (!input) return;
  const show = input.type === 'password';
  input.type = show ? 'text' : 'password';
  b.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
  b.classList.toggle('on', show);
});