// script.js

const input = document.getElementById("user-input");
const btn = document.getElementById("send-btn");
const grid = document.getElementById("movie-grid");
const title = document.getElementById("category-title");

// URL for our FastAPI backend
// This assumes the frontend and backend are running on the same machine
// Lightning AI automatically exposes port 8000
const API_URL = "https://lightning.ai/23d130/vision-model/studios/mysterious-cyan-vcip/web-ui?port%3D8000/recommend";
function renderRecommendations(recsList) {
  grid.innerHTML = "";
  if (recsList.length === 0) {
    title.textContent = "No recommendations found. Try a different query!";
    return;
  }
  
  recsList.forEach(rec => {
    const card = document.createElement("div");
    card.classList.add("movie-card");
    // We don't have posters, so we'll create a text-based card
    card.innerHTML = `
      <div class="movie-info">
        <h3>Item ASIN:</h3>
        <p style="color: #fff; font-size: 1rem;">${rec.asin}</p>
        <br>
        <h3>Predicted Rating:</h3>
        <p style="color: #e50914; font-size: 1.2rem; font-weight: 600;">${rec.rating}</p>
      </div>
    `;
    grid.appendChild(card);
  });
}

// Handle recommendation request
async function fetchRecommendations(query) {
  title.textContent = "Loading recommendations...";
  grid.innerHTML = ""; // Clear old results

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ reviews: query })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    title.textContent = `Music recommendations based on your movie taste:`;
    renderRecommendations(data.recommendations);

  } catch (error) {
    console.error("Error fetching recommendations:", error);
    title.textContent = "Error fetching recommendations. Is the backend running?";
  }
}

// Event Listeners
btn.addEventListener("click", () => {
  const query = input.value.trim();
  if (!query) return;
  fetchRecommendations(query);
  input.value = "";
});

input.addEventListener("keydown", e => {
  if (e.key === "Enter") btn.click();
});

// Netflix-style intro transition
window.addEventListener("load", () => {
  setTimeout(() => {
    document.getElementById("intro").style.display = "none";
    const mainContent = document.getElementById("main-content");
    mainContent.classList.remove("hidden");
    mainContent.style.opacity = "1";
  }, 5500); // after animation ends
});