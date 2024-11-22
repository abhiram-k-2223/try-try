// Handle the document upload form submission
const uploadForm = document.getElementById('upload-form');
uploadForm.addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent form from reloading the page

    const formData = new FormData(uploadForm);
    const uploadUrl = '/upload';  // The Flask endpoint for uploading documents

    // Send the form data (file) to the server using Fetch API
    fetch(uploadUrl, {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);  // Show success message
    })
    .catch(error => {
        alert('Error uploading document!');
        console.error('Error:', error);
    });
});

// Handle the query form submission
const queryForm = document.getElementById('query-form');
queryForm.addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent form from reloading the page

    const query = document.getElementById('query').value;
    const queryUrl = '/query';  // The Flask endpoint for querying the document

    // Send the query to the server using Fetch API
    fetch(queryUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
    })
    .then(response => response.json())
    .then(data => {
        const responseDiv = document.getElementById('response');
        if (data.answer) {
            responseDiv.innerHTML = `<strong>Answer:</strong> ${data.answer}`;
        } else {
            responseDiv.innerHTML = `<strong>Error:</strong> ${data.error || 'No answer found.'}`;
        }
        responseDiv.style.display = 'block';  // Show the response
    })
    .catch(error => {
        alert('Error querying document!');
        console.error('Error:', error);
    });
});
