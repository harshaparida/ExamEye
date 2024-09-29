// Get the examination button and container elements
const examinationButton = document.getElementById('examination-button');
const examinationContainer = document.getElementById('examination-container');

// Add an event listener to the examination button
examinationButton.addEventListener('click', () => {
  // Toggle the visibility of the examination container
  examinationContainer.style.display = examinationContainer.style.display === 'block' ? 'none' : 'block';
});