// logout.js
const logoutButton = document.getElementById('logout-button');

logoutButton.addEventListener('click', () => {
  fetch('/logout', {
    method: 'POST'
  })
  .then((response) => response.json())
  .then((data) => {
    if (data.success) {
      // Clear cache
      window.location.reload(true);
      // Redirect to login page
      window.location.href = '/';
    } else {
      alert('Failed to log out');
    }
  })
  .catch((error) => console.error(error));
});

