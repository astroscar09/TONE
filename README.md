# TONE

# Summary

TONE is a Lyman-alpha spectroscopic survey that uses the VIRUS instrument on the Hobbey-Ebberly Telescope (HET). The main goal of TONE is to detect Lyman-alpha from thousands of galaxies so that we can use these galaxies to map their galaxy properties to emerged Lyman-alpha emission. I lead the analysis of this project that took in the spectra from VIRUS and crossed matched the spectroscopic data to photometric data where there was overlap. The repository here stores the code that generated the data for the Lyra code, a Python code that can take in galaxy properties and returns back the posterior estimates of the emerged Lyman-alpha strength.

# Breakdown of the Code

1. Cross-match all the photometric catalogs (6 Fields) to the spectroscopic database (> 11 million sources)
2. Fit the Phomotetry with SED fitting code to get Galaxy Properties
3. Determine Lyman-alpha confidence
4. Validation and Testing 
5. Merging the data into an ML ready training set to be used in Lyra

