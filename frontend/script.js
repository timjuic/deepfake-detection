document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("input");
  input.addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        input.disabled = true;
        const image = document.getElementById("image");
        const response = document.getElementById("response");
        image.src = e.target.result;
        image.style.display = "block";

        const formData = new FormData();
        formData.append('image', file); 

        fetch("/upload-image", {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((data) => {
            response.textContent = data.message;
            response.style.display = "inline-block";

            if (response.textContent == "Real") {
              response.style.color = "#50C878";
            } else if (response.textContent == "DeepFake") {
              response.style.color = "#D21F3C";
            } else {
              response.style.color = "#8B0000";
            }

            input.disabled = false;
            })
            .catch(error => {
                console.error('Error:', error);
                response.textContent = 'Greška prilikom komunikacije s poslužiteljem';
                response.style.color = '#8B0000';
                response.style.display = 'inline-block';
                input.disabled = false;
            });
      };
      reader.readAsDataURL(file);
    }
  });
});