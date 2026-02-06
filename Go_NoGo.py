"""
GO/NO-GO TASK - Versão Simplificada
Execute este arquivo diretamente
"""
from psychopy import visual, core, event, data
import random
import pandas as pd
import os

# Janela (modo janela para teste)
win = visual.Window([800, 600], color='gray', fullscr=False)
mouse = event.Mouse(win=win)

# Parâmetros ajustados
N_TRIALS = 30  # Apenas 30 trials para teste
GO_PROB = 0.7  # 70% GO, 30% NOGO

# CRIAR COMPONENTES
# Tela de boas-vindas
welcome_text = visual.TextStim(win, 
    text="BEM-VINDO À TAREFA GO/ NO GO, Prima ESPAÇO para continuar",
    color='black', height=0.06, wrapWidth=1.5)

# Instruções
instructions = visual.TextStim(win,
    text="""INSTRUÇÕES:

círculo VERDE → Primir ESPAÇO 
círculo VERMELHO → NÃO primir nada

Terá 1 segundo para responder

Prima ESPAÇO para começar""",
    color='black', height=0.05, wrapWidth=1.8)

# Estímulos
fixation = visual.TextStim(win, text="+", color='black', height=0.1)
go_circle = visual.Circle(win, radius=0.15, fillColor='green', lineColor='green')
nogo_circle = visual.Circle(win, radius=0.15, fillColor='red', lineColor='red')
feedback = visual.TextStim(win, text="", color='black', height=0.07)

# TELA INICIAL 
def show_screen(text_stim, wait_key='space'):
    """Mostra uma tela e espera por tecla"""
    text_stim.draw()
    win.flip()
    if wait_key:
        event.waitKeys(keyList=[wait_key, 'escape'])
    return True

# Mostrar tela inicial
show_screen(welcome_text)

# Mostrar instruções
show_screen(instructions)

#BLOCO DE PRÁTICA 
print("\n📝 Iniciando bloco de práctica...")
practice_trials = [
    {'type': 'go', 'color': 'green'},
    {'type': 'nogo', 'color': 'red'},
    {'type': 'go', 'color': 'green'},
    {'type': 'nogo', 'color': 'red'}
]

practice_results = []

for i, trial in enumerate(practice_trials):
    print(f"  Práctica {i+1}/{len(practice_trials)}: {trial['type']}")
    
    # Fixação
    fixation.draw()
    win.flip()
    core.wait(0.8)
    
    # Estímulo
    if trial['type'] == 'go':
        stim = go_circle
        correct_key = 'space'
    else:
        stim = nogo_circle
        correct_key = None
    
    stim.draw()
    win.flip()
    
    # Coletar resposta (1 segundo)
    clock = core.Clock()
    keys = event.waitKeys(maxWait=1.0, keyList=['space', 'escape'], 
                         timeStamped=clock, clearEvents=True)
    
    # Processar resposta
    response = None
    rt = None
    
    if keys:
        key, rt = keys[0]
        if key == 'escape':
            print("✗ Experimento cancelado")
            win.close()
            core.quit()
        response = key
    
    # Feedback
    if trial['type'] == 'go':
        if response == 'space':
            feedback.text = "✓ Correto!"
            feedback.color = 'green'
        else:
            feedback.text = "✗ Demasiado lento"
            feedback.color = 'red'
    else:  # NOGO
        if response is None:
            feedback.text = "✓ Correcto!"
            feedback.color = 'green'
        else:
            feedback.text = "✗ Erro:Não devia primir "
            feedback.color = 'red'
    
    feedback.draw()
    win.flip()
    core.wait(0.8)
    
    # Intervalo
    win.flip()
    core.wait(0.5)

# TELA DE TRANSIÇÃO
transition = visual.TextStim(win,
    text="Agora vai começar a experiência real.\n\nPrima ESPAÇO para começar",
    color='black', height=0.05)
show_screen(transition)

#EXPERIÊNCIA REAL
print("\n A iniciar a experiêcnia real")
results = []

# Gerar lista de trials
all_trials = []
for i in range(N_TRIALS):
    if random.random() < GO_PROB:
        all_trials.append({'type': 'go', 'color': 'green'})
    else:
        all_trials.append({'type': 'nogo', 'color': 'red'})

random.shuffle(all_trials)

# Loop principal
for trial_num, trial in enumerate(all_trials, 1):
    print(f"  Trial {trial_num}/{N_TRIALS}: {trial['type']}")
    
    # Fixação com jitter
    fixation.draw()
    win.flip()
    core.wait(0.5 + random.uniform(-0.2, 0.2))
    
    # Estímulo
    if trial['type'] == 'go':
        stim = go_circle
    else:
        stim = nogo_circle
    
    stim.draw()
    win.flip()
    
    # Coletar resposta
    clock = core.Clock()
    keys = event.waitKeys(maxWait=1.0, keyList=['space', 'escape'], 
                         timeStamped=clock, clearEvents=True)
    
    response = None
    rt = None
    if keys:
        key, rt = keys[0]
        if key == 'escape':
            break
        response = key
    
    # Determinar acerto
    if trial['type'] == 'go':
        correct = 1 if response == 'space' else 0
    else:
        correct = 1 if response is None else 0
    
    # Feedback breve (apenas para erros)
    if trial['type'] == 'go' and response is None:
        #feedback.text = "Mais rápido!"
        #feedback.color = 'orange'
        #feedback.draw()
        win.flip()
        core.wait(0.3)
    elif trial['type'] == 'nogo' and response is not None:
        #feedback.text = "Não devia ter primido"
        #feedback.color = 'orange'
        #feedback.draw()
        win.flip()
        core.wait(0.3)
    
    # Salvar dados
    results.append({
        'trial': trial_num,
        'type': trial['type'],
        'response': response,
        'rt': rt,
        'correct': correct,
        'timestamp': data.getDateStr()
    })
    
    # Intervalo
    win.flip()
    core.wait(0.6 + random.uniform(-0.1, 0.1))

if results:
    # Salvar dados
    df = pd.DataFrame(results)
    
    # Criar pasta se não existir
    if not os.path.exists('resultados_go_nogo'):
        os.makedirs('resultados_go_nogo')
    
    # Nome do arquivo
    filename = f"resultados_go_nogo/participante_{data.getDateStr(format='%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f" Dados guardados em: {filename}")
    
    # Estatísticas
    total = len(df)
    hits = sum((df['type'] == 'go') & (df['correct'] == 1))
    misses = sum((df['type'] == 'go') & (df['correct'] == 0))
    correct_rejections = sum((df['type'] == 'nogo') & (df['correct'] == 1))
    false_alarms = sum((df['type'] == 'nogo') & (df['correct'] == 0))
    
    # Tela final
    final_text = f"""
    TAREFA COMPLETA
    
    Resultados:
    • Trials GO acertados: {hits}/{sum(df['type']=='go')}
    • Trials GO fallados: {misses}
    • Trials NOGO corretos: {correct_rejections}/{sum(df['type']=='nogo')}
    • Falsos alarmes: {false_alarms}
    
    Obrigada por participar.
    
    Primir ESPAÇO para sair
    """
    
    final_screen = visual.TextStim(win, text=final_text, 
                                  color='black', height=0.045, wrapWidth=1.8)
    show_screen(final_screen)

# Fechar
print("\n Experiência finalizada")
win.close()
core.quit()