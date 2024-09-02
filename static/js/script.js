export default function initPDFViewer(pdfUrl) {
    let pdfDoc = null,
        pageNum = 1,
        canvas = document.getElementById('pdf-render'),
        ctx = canvas.getContext('2d');

    // Render the PDF
    function renderPage(num) {
        pdfDoc.getPage(num).then(function(page) {
            const viewport = page.getViewport({ scale: 1.5 });
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            const renderCtx = {
                canvasContext: ctx,
                viewport: viewport
            };
            page.render(renderCtx).promise.then(() => {
                // Handle text selection
                canvas.addEventListener('mouseup', handleTextSelection);
            });
        });
    }

    pdfjsLib.getDocument(pdfUrl).promise.then(function(pdfDoc_) {
        pdfDoc = pdfDoc_;
        renderPage(pageNum);
    });

    function handleTextSelection() {
        const selectedText = window.getSelection().toString().trim();
        if (selectedText.length > 0) {
            const rect = window.getSelection().getRangeAt(0).getBoundingClientRect();
            const containerRect = canvas.getBoundingClientRect();
            
            // Position the annotation box near the selected text
            const annotationBox = document.getElementById('annotation-box');
            annotationBox.style.top = `${rect.bottom + window.scrollY}px`;
            annotationBox.style.left = `${rect.left + window.scrollX}px`;
            annotationBox.style.display = 'block';

            // Save the position and selected text for later use
            annotationBox.dataset.selectedText = selectedText;
            annotationBox.dataset.pageNum = pageNum;
        }
    }

    document.getElementById('save-note').addEventListener('click', function() {
        const note = document.getElementById('note-input').value;
        const annotationBox = document.getElementById('annotation-box');
        const selectedText = annotationBox.dataset.selectedText;
        const pageNum = annotationBox.dataset.pageNum;

        if (selectedText) {
            saveAnnotation(selectedText, note, pageNum);

            // Hide the annotation box after saving
            annotationBox.style.display = 'none';
            document.getElementById('note-input').value = '';
        }
    });

    function saveAnnotation(highlightText, note, pageNum) {
        const data = {
            filename: window.location.href.split('/').pop(),
            page_number: pageNum,
            highlight_text: highlightText,
            note: note
        };

        fetch('/save_annotation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        }).then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Annotation saved!');
            }
        });
    }
}
