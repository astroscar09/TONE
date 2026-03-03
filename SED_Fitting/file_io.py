from astropy.table import Table

def read_astropy_table(file):       
    tab = Table.read(file)

    return tab
