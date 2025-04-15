document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("movie");
  const suggestions = document.getElementById("suggestions");
  
  let csvData = null;
  let titlesAndYears = [];
  let debounceTimer;

  // Fetch CSV data once when page loads
  fetch("../movies.csv")
    .then((response) => response.text())
    .then((data) => {
      csvData = data;
      titlesAndYears = parseCSV(csvData);
    })
    .catch((error) => {
      console.error("Error fetching Netflix titles:", error);
    });

  input.addEventListener("input", function () {
    // Clear previous timer
    clearTimeout(debounceTimer);
    
    // Set a new timer to delay processing
    debounceTimer = setTimeout(() => {
      const inputValue = input.value.trim().toLowerCase();
      
      // Don't clear suggestions immediately - better UX
      if (inputValue.length < 3) {
        suggestions.innerHTML = "";
        return;
      }

      // Don't refetch data - use cached data
      if (titlesAndYears.length > 0) {
        const filteredTitles = titlesAndYears.filter((item) =>
          item.title.toLowerCase().includes(inputValue)
        );

        // Use CSS transitions for smooth appearance
        suggestions.style.opacity = "0";
        setTimeout(() => {
          // Clear and repopulate
          suggestions.innerHTML = "";
          
          filteredTitles.slice(0, 8).forEach((item) => {
            const li = document.createElement("li");
            li.textContent = `${item.title} (${item.year})`;
            suggestions.appendChild(li);
          });
          
          suggestions.style.opacity = "1";
        }, 150);
      }
    }, 300); // 300ms delay helps reduce flickering
  });

  suggestions.addEventListener("click", function (event) {
    if (event.target.tagName === "LI") {
      const selectedTitle = event.target.textContent.split(" (")[0];
      input.value = selectedTitle;
      suggestions.innerHTML = "";
    }
  });

  input.addEventListener("input", function () {
    input.classList.remove("selected");
  });

  // Handle keyboard navigation
  input.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (suggestions.firstChild) {
        suggestions.firstChild.focus();
        suggestions.firstChild.classList.add('focused');
      }
    }
  });

  suggestions.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (document.activeElement.nextElementSibling) {
        document.activeElement.classList.remove('focused');
        document.activeElement.nextElementSibling.focus();
        document.activeElement.classList.add('focused');
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (document.activeElement.previousElementSibling) {
        document.activeElement.classList.remove('focused');
        document.activeElement.previousElementSibling.focus();
        document.activeElement.classList.add('focused');
      } else {
        document.activeElement.classList.remove('focused');
        input.focus();
      }
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (document.activeElement.tagName === 'LI') {
        input.value = document.activeElement.textContent.split(" (")[0];
        suggestions.innerHTML = "";
        input.focus();
      }
    }
  });

  function parseCSV(csvData) {
    const lines = csvData.split("\n");
    const titlesAndYears = [];
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line) {
        const columns = line.split(",");
        const title = columns[1];
        const year = columns[3];
        titlesAndYears.push({ title, year });
      }
    }
    return titlesAndYears;
  }
});