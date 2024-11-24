document.addEventListener('DOMContentLoaded', function() {
    const uploadSection = document.getElementById('uploadSection');
    const chatSection = document.getElementById('chatSection');
    const questionInput = document.getElementById('questionInput');
    const charCount = document.getElementById('charCount');
    const uploadForm = document.getElementById('uploadForm');
    const chatForm = document.getElementById('chatForm');
    const chatMessages = document.getElementById('chatMessages');

    if (questionInput && charCount) {
        questionInput.addEventListener('input', () => {
            const length = questionInput.value.length;
            charCount.textContent = `${length}/500`;
            if (length > 500) {
                charCount.classList.add('text-red-500');
            } else {
                charCount.classList.remove('text-red-500');
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            const fileInput = document.getElementById('pdfFile');
            const uploadStatus = document.getElementById('uploadStatus');
            
            if (!fileInput.files[0]) {
                uploadStatus.innerHTML = '<p class="text-red-500">Please select a file first</p>';
                return;
            }
            
            formData.append('file', fileInput.files[0]);
            uploadStatus.innerHTML = `
                <p class="text-blue-500">Processing your study material...</p>
                <div class="mt-2 w-full h-2 bg-gray-200 rounded-full">
                    <div class="upload-progress-bar h-full rounded-full"></div>
                </div>
            `;
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.success) {
                    uploadStatus.innerHTML = `
                        <p class="text-green-500">${data.message}</p>
                        <p class="text-sm text-gray-600 mt-2">You can now ask questions about your study material!</p>
                    `;
                    uploadSection.classList.add('opacity-0');
                    setTimeout(() => {
                        uploadSection.classList.add('hidden');
                        chatSection.classList.remove('hidden');
                        chatSection.classList.add('opacity-0');
                        setTimeout(() => {
                            chatSection.classList.remove('opacity-0');
                            addMessage('assistant', 'Hi! I\'ve processed your study material. What would you like to know about it?');
                        }, 50);
                    }, 300);
                } else {
                    uploadStatus.innerHTML = `<p class="text-red-500">Error: ${data.error}</p>`;
                }
            } catch (error) {
                uploadStatus.innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
            }
        });
    }

    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = questionInput.value.trim();
            
            if (!question) return;
            if (question.length > 500) {
                alert('Please keep your question under 500 characters');
                return;
            }
            
            addMessage('user', question);
            questionInput.value = '';
            charCount.textContent = '0/500';
            
            const thinking = addMessage('assistant', '🤔 Thinking...');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                
                thinking.remove();
                
                if (data.success) {
                    addMessage('assistant', data.answer);
                } else {
                    addMessage('error', data.error || 'An error occurred');
                }
            } catch (error) {
                thinking.remove();
                addMessage('error', error.message);
            }
        });
    }
    
    function addMessage(type, content) {
        if (!chatMessages) return null;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `p-4 rounded-lg message-enter ${
            type === 'user' ? 'bg-blue-100 ml-12' :
            type === 'assistant' ? 'bg-green-100 mr-12' :
            'bg-red-100'
        } mb-4`;
        
        messageDiv.innerHTML = `
            <p class="font-semibold mb-1">${
                type === 'user' ? 'You' :
                type === 'assistant' ? 'Study Buddy' :
                'Error'
            }</p>
            <p class="text-gray-700">${content}</p>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        requestAnimationFrame(() => {
            messageDiv.classList.remove('message-enter');
            messageDiv.classList.add('message-enter-active');
        });
        
        return messageDiv;
    }
});