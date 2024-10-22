function updateWordCount() {
  const abstractInput = document.querySelector(".form-control-abstract");
  const wordCountElement = document.querySelector(".word-count");
  const words = abstractInput.value.trim().split(/\s+/).filter(Boolean);
  wordCountElement.textContent = `Word Count: ${words.length}/200`;
}
