import sys as _sys # For system output

def update_progress(prog):
    """
    Progress bar:

    - Adapted from: https://stackoverflow.com/questions/3160699/python-progress-bar
    """
    brl = 10; stat = '';
    if isinstance(prog, int): prog = float(prog)
    bar = int(round(brl * prog))
    txt = "\r [{0}] {1}% {2}".format(
        '#'*bar + '-'*(brl - bar),
        round(prog * 100), stat
    ); _sys.stdout.write(txt); _sys.stdout.flush();
