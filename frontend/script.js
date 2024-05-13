document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('input').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const image = document.getElementById('image');
                image.src = e.target.result;
                image.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
});
