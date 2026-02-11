"""
GO/NO-GO TASK 
"""
from psychopy import visual, core, event, data
import random
import pandas as pd
import os
import sys

def run_gonogo_task(win=None):
    if win is None:
        win = visual.Window([900, 800], color='gray', fullscr=False)

    # Parameters
    N_TRIALS = 100
    GO_PROB = 0.75  # 75% GO trials, 25% NOGO trials
    
    IMAGE_FOLDER = "go_nogo_stim"      
    
    def show_screen(text_stim, wait_key='space'):
        """Mostra uma tela e espera por tecla"""
        text_stim.draw()
        win.flip()
        if wait_key:
            event.waitKeys(keyList=[wait_key, 'escape'])
        return True
    
    welcome_text = visual.TextStim(win, 
        text="BEM-VINDO À TAREFA GO/NO GO\n\nPrima ESPAÇO para continuar",
        color='white', height=0.1, wrapWidth=1.75)
    
    fixation = visual.TextStim(win, text="+", color='white', height=0.25) 
    feedback = visual.TextStim(win, text="", color='white', height=0.07)
    
    GO_IMAGES = [
        {
            'name': 'Go_stim_1',
            'filename': 'Go_stim_1.png', 
            'description': 'Cruz'
        },
        {
            'name': 'Go_stim_2', 
            'filename': 'Go_stim_2.png',
            'description': 'Barra lateral para a esquerda'
        },
        {
            'name': 'Go_stim_3',
            'filename': 'Go_stim_3.png',  
            'description': 'Barra lateral para a direita'
        },
        {
            'name': 'Go_stim_4',
            'filename': 'Go_stim_4.png',  
            'description': 'Barra vertical curta'
        }
    ]
    
    NOGO_IMAGES = [
        {
            'name': 'NoGo_stim',
            'filename': 'NoGo_stim.png',  
            'description': 'Barra vertical longa'
        }
    ]

    # LOAD IMAGES
    go_stimuli = []
    nogo_stimuli = []
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"ERROR: Folder '{IMAGE_FOLDER}' not found!")
        print("Please create a folder named 'images' with your stimulus images.")
        print("Expected image files:")
        print("- Go_stim_1.png")
        print("- Go_stim_2.png")
        print("- Go_stim_3.png")
        print("- Go_stim_4.png")
        print("- NoGo_stim")
        return
    
    # Load Go images
    for go_img in GO_IMAGES:
        image_path = os.path.join(IMAGE_FOLDER, go_img['filename'])
        if os.path.exists(image_path):
            stim = visual.ImageStim(win, image=image_path, size=(1.25, 1.25))
            go_stimuli.append({
                'name': go_img['name'],
                'stim': stim,
                'filename': go_img['filename'],
                'description': go_img['description']
            })
            print(f" Loaded: {go_img['filename']}")
        else:
            print(f" ERROR: Image not found: {image_path}")
            print(f"  Please add this image to the '{IMAGE_FOLDER}' folder")
            # Create a placeholder instead
            stim = visual.Rect(win, width=0.3, height=0.3, fillColor='gray')
            go_stimuli.append({
                'name': go_img['name'],
                'stim': stim,
                'filename': go_img['filename'],
                'description': go_img['description'] + " (MISSING)"
            })
    
    # Load No-Go image
    for nogo_img in NOGO_IMAGES:
        image_path = os.path.join(IMAGE_FOLDER, nogo_img['filename'])
        if os.path.exists(image_path):
            stim = visual.ImageStim(win, image=image_path, size=(1.25, 1.25))
            nogo_stimuli.append({
                'name': nogo_img['name'],
                'stim': stim,
                'filename': nogo_img['filename'],
                'description': nogo_img['description']
            })
            print(f" Loaded: {nogo_img['filename']}")
        else:
            print(f"ERROR: Image not found: {image_path}")
            print(f"  Please add this image to the '{IMAGE_FOLDER}' folder")
            stim = visual.Rect(win, width=0.3, height=0.3, fillColor='gray')
            nogo_stimuli.append({
                'name': nogo_img['name'],
                'stim': stim,
                'filename': nogo_img['filename'],
                'description': nogo_img['description'] + " (MISSING)"
            })
    
    # TELA INICIAL 
    def show_instructions_with_images():
        """Mostra as instruções e espera por tecla"""
        win.flip()
        
       # Título
        title = visual.TextStim(win, text="INSTRUÇÕES", 
                               color='white', height=0.06, pos=(0, 0.4))
        title.draw()
        
        # Instrução para GO
        go_text = visual.TextStim(win, 
            text="PRIMA ESPAÇO quando vir:", 
            color='white', height=0.05, pos=(0, 0.3))
        go_text.draw()
        
        # Mostrar as 4 imagens GO em linha
        go_positions = [(-0.4, 0.15), (-0.13, 0.15), (0.13, 0.15), (0.4, 0.15)]
        for i, go_stim in enumerate(go_stimuli):
            img_path = os.path.join(IMAGE_FOLDER, go_stim['filename'])
            if os.path.exists(img_path):
                img = visual.ImageStim(win, image=img_path, 
                                       size=(0.25, 0.25), pos=go_positions[i])
                img.draw()
        
        # Instrução para NOGO
        nogo_text = visual.TextStim(win, 
            text="NÃO PRIMA quando vir:", 
            color='white', height=0.05, pos=(0, -0.05))
        nogo_text.draw()
        
        # Mostrar a imagem NOGO sozinha
        if nogo_stimuli and os.path.exists(os.path.join(IMAGE_FOLDER, nogo_stimuli[0]['filename'])):
            nogo_img = visual.ImageStim(win, 
                image=os.path.join(IMAGE_FOLDER, nogo_stimuli[0]['filename']), 
                size=(0.25, 0.25), pos=(0, -0.2))
            nogo_img.draw()
        
        # Instrução final
        final_text = visual.TextStim(win,
            text="Terá 1 segundo para responder\n\nPrima ESPAÇO para começar", 
            color='white', height=0.045, pos=(0, -0.40))
        final_text.draw()
        
        win.flip()
        event.waitKeys(keyList=['space', 'escape'])

    # Mostrar tela inicial
    show_screen(welcome_text)

    # Mostrar instruções
    show_instructions_with_images()

    # BLOCO DE PRÁTICA 
    print("\n A iniciar exemplo...")
    practice_trials = [
        {'type': 'go', 'stim_index': 0},    
        {'type': 'nogo', 'stim_index': 0},   
        {'type': 'go', 'stim_index': 1},    
        {'type': 'go', 'stim_index': 2},    
        {'type': 'nogo', 'stim_index': 0},   
        {'type': 'go', 'stim_index': 3}  
    ]

    event.clearEvents()
    for i, trial in enumerate(practice_trials):
        if trial['type'] == 'go':
            stim_data = go_stimuli[trial['stim_index']]
            stim = stim_data['stim']
            stim_name = stim_data['description']
        else:
            stim_data = nogo_stimuli[trial['stim_index']]
            stim = stim_data['stim']
            stim_name = stim_data['description']
        
        print(f"  Práctica {i+1}/{len(practice_trials)}: {trial['type']} - {stim_name}")
        
        # Fixação
        fixation.draw()
        win.flip()
        core.wait(0.8)
        
        # Estímulo
        stim.draw()
        win.flip()
        
        # Ter resposta (1 segundo)
        clock = core.Clock()
        keys = event.waitKeys(maxWait=1.0, keyList=['space', 'escape'], 
                             timeStamped=clock, clearEvents=True)
        
        # Processar resposta
        response = None
        rt = None
        
        if keys:
            key, rt = keys[0]
            if key == 'escape':
                print(" Experiência cancelada")
                return None
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
                feedback.text = "✓ Correto!"
                feedback.color = 'green'
            else:
                feedback.text = "✗ Erro: Não devia primir"
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
        color='white', height=0.05)
    show_screen(transition)

    # EXPERIÊNCIA REAL
    print("\n A iniciar a experiência real")
    results = []

    # Gerar lista de trials
    all_trials = []
    
    # Primeiros 10 trials - sempre Go (distribuído entre os 3 tipos)
    for i in range(10):
        go_type = i % len(go_stimuli)  # Alterna entre os tipos de Go
        all_trials.append({
            'type': 'go', 
            'stim_index': go_type,
            'stim_name': go_stimuli[go_type]['name'],
            'description': go_stimuli[go_type]['description'],
            'filename': go_stimuli[go_type]['filename']
        })
    
    # Restantes trials - aleatórios com probabilidade GO_PROB
    for i in range(10, N_TRIALS):
        if random.random() < GO_PROB:
            # Go trial - escolher aleatoriamente um dos tipos
            go_type = random.randint(0, len(go_stimuli) - 1)
            all_trials.append({
                'type': 'go', 
                'stim_index': go_type,
                'stim_name': go_stimuli[go_type]['name'],
                'description': go_stimuli[go_type]['description'],
                'filename': go_stimuli[go_type]['filename']
            })
        else:
            # No-Go trial
            all_trials.append({
                'type': 'nogo',
                'stim_index': 0,
                'stim_name': nogo_stimuli[0]['name'],
                'description': nogo_stimuli[0]['description'],
                'filename': nogo_stimuli[0]['filename']
            })
    
    # Randomizar trials (exceto os primeiros 10 que são sempre Go)
    remaining_trials = all_trials[10:]
    random.shuffle(remaining_trials)
    all_trials = all_trials[:10] + remaining_trials

    # Loop principal
    event.clearEvents()
    for trial_num, trial in enumerate(all_trials, 1):
        print(f"  Trial {trial_num}/{N_TRIALS}: {trial['type']} - {trial['description']}")
        
        # Fixação com jitter
        fixation.draw()
        win.flip()
        core.wait(0.5 + random.uniform(-0.2, 0.2))
        
        # Selecionar estímulo
        if trial['type'] == 'go':
            stim = go_stimuli[trial['stim_index']]['stim']
        else:
            stim = nogo_stimuli[trial['stim_index']]['stim']
        
        # Apresentar estímulo
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
        
        # Salvar dados (incluindo nome do arquivo)
        results.append({
            'trial': trial_num,
            'trial_type': trial['type'],  # 'go' ou 'nogo'
            'stimulus_type': trial['stim_name'],  # Nome do estímulo
            'stimulus_description': trial['description'],  # Descrição
            'image_filename': trial['filename'],  # Nome do arquivo da imagem
            'go_type_index': trial['stim_index'] if trial['type'] == 'go' else None,
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
        
        # Estatísticas detalhadas
        total_go = df[df['trial_type'] == 'go'].shape[0]
        total_nogo = df[df['trial_type'] == 'nogo'].shape[0]
        
        # Estatísticas por tipo de Go
        go_stats = {}
        for go_type in ['Go_stim_1', 'Go_stim_2', 'Go_stim_3', 'Go_stim_4']:
            go_trials = df[df['stimulus_type'] == go_type]
            if len(go_trials) > 0:
                go_stats[go_type] = {
                    'hits': go_trials['correct'].sum(),
                    'total': len(go_trials),
                    'accuracy': go_trials['correct'].mean() * 100
                }
        
        # Estatísticas gerais
        hits = df[(df['trial_type'] == 'go') & (df['correct'] == 1)].shape[0]
        misses = df[(df['trial_type'] == 'go') & (df['correct'] == 0)].shape[0]
        correct_rejections = df[(df['trial_type'] == 'nogo') & (df['correct'] == 1)].shape[0]
        false_alarms = df[(df['trial_type'] == 'nogo') & (df['correct'] == 0)].shape[0]
        
        # Calcular tempo médio de reação para Go trials corretos
        correct_go_trials = df[(df['trial_type'] == 'go') & (df['correct'] == 1)]
        correct_go_rt = correct_go_trials['rt'].mean() if not correct_go_trials.empty else 0
        
        # Tela final com estatísticas detalhadas
        final_text = f"""
        TAREFA COMPLETA
        
        Resultados Gerais:
        • Trials GO acertados: {hits}/{total_go} ({hits/total_go*100:.1f}%)
        • Trials GO falhados: {misses}
        • Trials NOGO corretos: {correct_rejections}/{total_nogo} ({correct_rejections/total_nogo*100:.1f}%)
        • Falsos alarmes: {false_alarms}
        • Tempo médio reação (Go): {correct_go_rt:.3f}s
        
        Resultados por Tipo de Go:
        • Círculo Verde: {go_stats.get('go_circle', {}).get('hits', 0)}/{go_stats.get('go_circle', {}).get('total', 0)} acertos
        • Quadrado Azul: {go_stats.get('go_square', {}).get('hits', 0)}/{go_stats.get('go_square', {}).get('total', 0)} acertos
        • Triângulo Amarelo: {go_stats.get('go_triangle', {}).get('hits', 0)}/{go_stats.get('go_triangle', {}).get('total', 0)} acertos
        
        Obrigada por participar.
        
        Prima ESPAÇO para sair
        """
        
        final_screen = visual.TextStim(win, text=final_text, 
                                      color='black', height=0.04, wrapWidth=1.8)
        show_screen(final_screen)

    # Fechar
    print("\n Experiência finalizada")
    return results