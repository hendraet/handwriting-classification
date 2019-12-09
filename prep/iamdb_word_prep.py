with open('numbers_s.txt', 'r') as f, open('numbers_clean.txt', 'a') as out:
    for line in f:
        parts = line.split(' ')
        replacements = [('pt', '.'), ('-', ''), ('cm', ','), ('s_', '')]
        cleansed_string = parts[1]
        for replacement in replacements:
            cleansed_string = cleansed_string.replace(replacement[0], replacement[1])
        print(parts[0], cleansed_string)
        out.write(parts[0] + ' ' + cleansed_string)
