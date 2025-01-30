document.addEventListener("DOMContentLoaded", function () {
    fetch("/chatbot") // Adjust the path based on your project structure
        .then(response => response.text())
        .then(html => {
            document.getElementById("chatbot-container").innerHTML = html;
        })
        .catch(error => console.error("Error loading chatbot:", error));
});
function toggleChat() {
    const chatbox = document.getElementById("chatbox");
    chatbox.classList.toggle("show");
}



function sendMessage() {
    const userInput = document.getElementById("userInput");
    const chatBody = document.getElementById("chatBody");
    
    if (userInput.value.trim() === "") return;
    
    const userContainer = document.createElement("div");
    userContainer.classList.add("message-container");
    
    const userIcon = document.createElement("div");
    userIcon.classList.add("user-icon");
    
    const userMessage = document.createElement("div");
    userMessage.classList.add("message", "user-message");
    userMessage.textContent = userInput.value;
    
    userContainer.appendChild(userMessage);
    userContainer.appendChild(userIcon);
    chatBody.appendChild(userContainer);
    
    setTimeout(() => {
        const botContainer = document.createElement("div");
        botContainer.classList.add("message-container");
        
        const botIcon = document.createElement("div");
        botIcon.classList.add("bot-icon");
        
        const botMessage = document.createElement("div");
        botMessage.classList.add("message", "bot-message");
        botMessage.textContent = "Hello! How can I assist you?";
        
        botContainer.appendChild(botIcon);
        botContainer.appendChild(botMessage);
        chatBody.appendChild(botContainer);
    }, 1000);
    
    userInput.value = "";
    chatBody.scrollTop = chatBody.scrollHeight;
}
