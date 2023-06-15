import fileinput
import os

try:
    projection_path = os.path.join(os.path.dirname(
        __file__), '../fd_data/projections.csv')
    with fileinput.FileInput(projection_path, inplace=True) as file:
        for line in file:
            print(line.replace('III', '')
                  .replace('II', '')
                  .replace('IV', '')
                  .replace('Jr.', '')
                  .replace('Sr.', '')
                  .replace('Moe Harkless', 'Maurice Harkless')
                  .replace('PJ Washington', 'P.J. Washington')
                  .replace('Jakarr Sampson', 'JaKarr Sampson')
                  .replace('KJ Martin', 'Kenyon Martin')
                  .replace('Cameron Thomas', 'Cam Thomas')
                  .replace('Nicolas Claxton', 'Nic Claxton')
                  .replace('Marjon Beauchamp', 'MarJon Beauchamp')
                  .replace('DeAndre Ayton', 'Deandre Ayton')
                  .replace('Guillermo Hernangomez', 'Willy Hernangomez')
                  .replace(' \"', '\"'), end='')
except:
    print('FD Projections failed to rename')
    pass

try:
    ownership_path = os.path.join(os.path.dirname(
        __file__), '../fd_data/ownership.csv')
    with fileinput.FileInput(ownership_path, inplace=True) as file:
        for line in file:
            print(line.replace('III', '')
                  .replace('II', '')
                  .replace('IV', '')
                  .replace('Jr.', '')
                  .replace('Sr.', '')
                  .replace('Moe Harkless', 'Maurice Harkless')
                  .replace('PJ Washington', 'P.J. Washington')
                  .replace('Jakarr Sampson', 'JaKarr Sampson')
                  .replace('KJ Martin', 'Kenyon Martin')
                  .replace('Cameron Thomas', 'Cam Thomas')
                  .replace('Nicolas Claxton', 'Nic Claxton')
                  .replace('Marjon Beauchamp', 'MarJon Beauchamp')
                  .replace('DeAndre Ayton', 'Deandre Ayton')
                  .replace('Guillermo Hernangomez', 'Willy Hernangomez')
                  .replace(' \"', '\"'), end='')
except:
    print('FD Ownership failed to rename')
    pass

try:
    ownership_path = os.path.join(os.path.dirname(
        __file__), '../fd_data/boom_bust.csv')
    with fileinput.FileInput(ownership_path, inplace=True) as file:
        for line in file:
            print(line.replace('III', '')
                  .replace('II', '')
                  .replace('IV', '')
                  .replace('Jr.', '')
                  .replace('Sr.', '')
                  .replace('Moe Harkless', 'Maurice Harkless')
                  .replace('PJ Washington', 'P.J. Washington')
                  .replace('Jakarr Sampson', 'JaKarr Sampson')
                  .replace('KJ Martin', 'Kenyon Martin')
                  .replace('Cameron Thomas', 'Cam Thomas')
                  .replace('Nicolas Claxton', 'Nic Claxton')
                  .replace('Marjon Beauchamp', 'MarJon Beauchamp')
                  .replace('DeAndre Ayton', 'Deandre Ayton')
                  .replace('Guillermo Hernangomez', 'Willy Hernangomez')
                  .replace(' \"', '\"'), end='')
except:
    print('FD Boom/Bust failed to rename')
    pass

try:
    projection_path = os.path.join(os.path.dirname(
        __file__), '../dk_data/projections.csv')
    with fileinput.FileInput(projection_path, inplace=True) as file:
        for line in file:
            print(line.replace('Matt Fitzpatrick', 'Matthew Fitzpatrick').replace(
                'J.T. Poston', 'JT Poston'), end='')
except:
    print('DK Projections failed to rename')
    pass

try:
    ownership_path = os.path.join(os.path.dirname(
        __file__), '../dk_data/ownership.csv')
    with fileinput.FileInput(ownership_path, inplace=True) as file:
        for line in file:
            print(line.replace('Matt Fitzpatrick', 'Matthew Fitzpatrick').replace(
                'J.T. Poston', 'JT Poston'), end='')
except:
    print('DK Ownership failed to rename')
    pass

try:
    ownership_path = os.path.join(os.path.dirname(
        __file__), '../dk_data/boom_bust.csv')
    with fileinput.FileInput(ownership_path, inplace=True) as file:
        for line in file:
            print(line.replace('Matt Fitzpatrick', 'Matthew Fitzpatrick').replace(
                'J.T. Poston', 'JT Poston'), end='')
except:
    print('DK Boom/Bust failed to rename')
    pass

try:
    lineups = os.path.join(os.path.dirname(
        __file__), '../dk_data/tournament_lineups.csv')
    with fileinput.FileInput(ownership_path, inplace=True) as file:
        for line in file:
            print(line.replace('Matt Fitzpatrick', 'Matthew Fitzpatrick').replace(
                'J.T. Poston', 'JT Poston'), end='')
except:
    print('DK Tournament Lineups failed to rename')
    pass

try:
    player_ids_path = os.path.join(os.path.dirname(
        __file__), '../dk_data/player_ids.csv')
    with fileinput.FileInput(player_ids_path, inplace=True) as file:
        for line in file:
            print(line.replace('Matt Fitzpatrick', 'Matthew Fitzpatrick').replace(
                'J.T. Poston', 'JT Poston'), end='')
except:
    print('DK Player IDs failed to rename')
    pass