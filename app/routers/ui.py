from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def interface_grafica():
    return """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MNIST Classifier - Interface</title>
        <style>
            :root {
                --primary: #4f46e5;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-muted: #64748b;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            body { background-color: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
            .container { background: var(--card-bg); padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05); width: 100%; max-width: 600px; text-align: center; }
            h1 { color: var(--text-main); font-size: 26px; margin-bottom: 8px; font-weight: 700; }
            .subtitle { color: var(--text-muted); font-size: 15px; margin-bottom: 30px; }
            
            .upload-label { display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 25px 20px; cursor: pointer; transition: all 0.2s ease; background-color: #f8fafc; margin-bottom: 25px; width: 100%; min-height: 180px; }
            .upload-label.highlight { border-color: var(--primary); background-color: #eef2ff; }
            .upload-label:hover { border-color: var(--primary); background-color: #eef2ff; }
            .upload-icon { font-size: 36px; margin-bottom: 10px; }
            .upload-text { color: var(--text-main); font-size: 16px; font-weight: 600; word-break: break-all; max-width: 90%; }
            .upload-subtext { color: var(--text-muted); font-size: 13px; margin-top: 5px; }
            
            .criteria-box { margin-top: 15px; padding: 12px 15px; background: rgba(255, 255, 255, 0.6); border: 1px solid #e2e8f0; border-radius: 8px; text-align: left; width: 85%; }
            .criteria-title { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
            .criteria-list { list-style-type: none; font-size: 12px; color: var(--text-muted); line-height: 1.5; }
            .criteria-list li::before { content: "•"; color: var(--primary); font-weight: bold; display: inline-block; width: 1em; }
            input[type="file"] { display: none; }

            .result-card { display: none; border-radius: 12px; padding: 24px; text-align: left; animation: fadeIn 0.4s ease-out; margin-bottom: 20px; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            .result-content { display: flex; gap: 20px; align-items: center; }
            .result-image-container { flex: 0 0 120px; background: white; border-radius: 8px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); display: flex; justify-content: center; align-items: center; }
            .result-image-container img { max-width: 100%; max-height: 100px; object-fit: contain; }
            .result-details { flex: 1; }
            .result-title { font-size: 20px; font-weight: 700; margin-bottom: 8px; }
            .result-message { font-size: 14px; margin-bottom: 15px; line-height: 1.4; opacity: 0.9; }
            
            .progress-container { margin-top: auto; }
            .progress-label { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; margin-bottom: 6px; }
            .progress-bar { background-color: rgba(0,0,0,0.1); border-radius: 999px; height: 8px; width: 100%; overflow: hidden; }
            .progress-fill { height: 100%; border-radius: 999px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); width: 0%; }

            /* Cores de Status */
            .status-valid { background-color: #ecfdf5; border: 1px solid #10b981; color: #065f46; }
            .status-valid .progress-fill { background-color: #10b981; }
            
            .status-ood { background-color: #fffbeb; border: 1px solid #f59e0b; color: #92400e; }
            .status-ood .progress-fill { background-color: #f59e0b; }

            /* NOVO: Cor de Erro (Imagem Inválida) */
            .status-error { background-color: #fef2f2; border: 1px solid #ef4444; color: #991b1b; }
            .status-error .progress-fill { display: none; } /* Esconde a barra de progresso */

            .loading { display: none; color: var(--primary); font-weight: 600; margin-top: 15px; }

            .btn-reset { display: none; width: 100%; padding: 14px; background-color: var(--primary); color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background-color 0.2s; margin-top: 10px; }
            .btn-reset:hover { background-color: #4338ca; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 MNIST Classifier</h1>
            <p class="subtitle">Faça o upload de uma imagem de um dígito (ou letra) para testar o modelo.</p>
            
            <label for="file-upload" class="upload-label" id="upload-area">
                <div class="upload-icon" id="upload-icon">📤</div>
                <div class="upload-text" id="upload-text">Clique ou arraste uma imagem</div>
                <div class="upload-subtext" id="upload-subtext">Formatos aceitos: JPG, PNG</div>
                
                <div class="criteria-box">
                    <div class="criteria-title">Critérios de Avaliação:</div>
                    <ul class="criteria-list">
                        <li>Imagem de um dígito manuscrito (0 a 9)</li>
                        <li>Fundo claro com traço escuro</li>
                        <li>Detecção de anomalias via Energy Score</li>
                    </ul>
                </div>

                <input id="file-upload" type="file" accept="image/*">
            </label>

            <div id="loading" class="loading">Analisando imagem...</div>

            <div id="result" class="result-card">
                <div class="result-content">
                    <div class="result-image-container">
                        <img id="result-img" src="" alt="Imagem">
                    </div>
                    <div class="result-details">
                        <div class="result-title" id="result-title"></div>
                        <div class="result-message" id="result-message"></div>
                        
                        <div class="progress-container" id="progress-container">
                            <div class="progress-label">
                                <span>Confiança do Modelo</span>
                                <span id="result-confidence-text"></span>
                            </div>
                            <div class="progress-bar">
                                <div id="result-progress" class="progress-fill"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <button id="reset-btn" class="btn-reset" onclick="resetApp()">Analisar outra imagem</button>
        </div>

        <script>
            const dropArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-upload');

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
            });

            dropArea.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) processFile(files[0]);
            }, false);

            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) processFile(this.files[0]);
            });

            function preventDefaults (e) { e.preventDefault(); e.stopPropagation(); }

            async function processFile(file) {
                if (!file.type.startsWith('image/')) {
                    alert("Por favor, envie apenas arquivos de imagem.");
                    return;
                }

                document.getElementById('reset-btn').style.display = "none";
                document.getElementById('upload-text').innerText = file.name;
                document.getElementById('upload-subtext').innerText = "Processando arquivo...";
                document.getElementById('upload-icon').innerText = "⏳";
                document.getElementById('loading').style.display = "block";
                document.getElementById('result').style.display = "none";

                const formData = new FormData();
                formData.append("file", file);

                try {
                    const response = await fetch("/v1/predict/image", {
                        method: "POST",
                        body: formData
                    });
                    
                    const data = await response.json();

                    if (!response.ok) {
                        // NOVO: Se a API retornou erro (ex: 422), joga a mensagem exata dela para o renderResult
                        throw new Error(data.detail || "Erro desconhecido na API.");
                    }
                    
                    renderResult(data, file);
                } catch (error) {
                    // NOVO: Renderiza o card de erro em vermelho em vez de dar alert genérico
                    renderError(error.message, file);
                } finally {
                    document.getElementById('loading').style.display = "none";
                }
            }

            function renderResult(data, file) {
                const card = document.getElementById('result');
                const img = document.getElementById('result-img');
                const title = document.getElementById('result-title');
                const message = document.getElementById('result-message');
                const confidenceText = document.getElementById('result-confidence-text');
                const progress = document.getElementById('result-progress');
                const progressContainer = document.getElementById('progress-container');
                const resetBtn = document.getElementById('reset-btn');

                img.src = URL.createObjectURL(file);
                title.innerText = "Previsão: " + data.predicted_class;
                message.innerText = data.message;
                
                const confPct = (data.confidence * 100).toFixed(1);
                confidenceText.innerText = confPct + "%";

                progressContainer.style.display = "block";
                card.className = "result-card";
                card.style.display = "block";
                progress.style.width = "0%";

                if (data.is_ood) {
                    card.classList.add("status-ood");
                    title.innerText = "⚠️ " + title.innerText;
                    document.getElementById('upload-icon').innerText = "⚠️";
                } else {
                    card.classList.add("status-valid");
                    title.innerText = "✅ " + title.innerText;
                    document.getElementById('upload-icon').innerText = "✅";
                }

                document.getElementById('upload-subtext').innerText = "Processamento concluído";
                resetBtn.style.display = "block";

                setTimeout(() => { progress.style.width = confPct + "%"; }, 100);
            }

            // NOVO: Função dedicada a mostrar erros (imagem em branco, etc) na tela
            function renderError(errorMessage, file) {
                const card = document.getElementById('result');
                const img = document.getElementById('result-img');
                const title = document.getElementById('result-title');
                const message = document.getElementById('result-message');
                const progressContainer = document.getElementById('progress-container');
                const resetBtn = document.getElementById('reset-btn');

                img.src = URL.createObjectURL(file);
                title.innerText = "❌ Imagem Inválida";
                message.innerText = errorMessage;
                progressContainer.style.display = "none"; // Esconde a barra de confiança

                card.className = "result-card status-error";
                card.style.display = "block";
                
                document.getElementById('upload-icon').innerText = "❌";
                document.getElementById('upload-subtext').innerText = "Falha no processamento";
                resetBtn.style.display = "block";
            }

            function resetApp() {
                fileInput.value = '';
                document.getElementById('upload-text').innerText = "Clique ou arraste uma imagem";
                document.getElementById('upload-subtext').innerText = "Formatos aceitos: JPG, PNG";
                document.getElementById('upload-icon').innerText = "📤";
                document.getElementById('result').style.display = "none";
                document.getElementById('reset-btn').style.display = "none";
            }
        </script>
    </body>
    </html>
    """