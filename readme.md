# featureExtractor.py
[github]( https://github.com/vidstrancar/flexible_feature_extractor)

## Značilke na voljo
Večina značilk je iz članka  [Enhancing energy efficiency in the residential sector with smart meter data analytics](https://link.springer.com/article/10.1007/s12525-018-0290-9) [Hopf, Sodenkamp, Staake, 2018].
Tabela se začne na strani 465.

 ### Seznam implementiranih značilk in dokumentacija:
 - list_features_implemented_and_display_doc.py
 
## Primeri uporabe
- V /data/consumption.csv imamo primer električne porabe enega gospodinjstva z granulacijo 15 minut za obdobje cca. 1, 5 leta. 
- poleg podatkov o porabi potrebujemo tudi podati za uro, ko se začne\konča jutro, dopoldne, poldne, popoldne, večer, noč, visokotarifno in nizkotarifen termin porabe energije (ht/nt)

-  ex1.py -- uporaba za ekstrakcijo vnaprej izbranih značilk: 'asc', 'skewness', 'kurtosis',  'MinMax', 'nonExistantFeature', 'k'  (ne potrebujejo podatkov o temperaturi). Po ekstrakciji izpišemo izračunane značilke in značilke, ki niso na voljo ter razloge zakaj niso na voljo ('nonExistantFeature' ni implementiran,  'k' oz. koeficient razmerja med porabo in zunanjo temperaturo ni navoljo, ker potrebuje podatke o temperaturi). 

- ex2.py -- uporaba za ekstrakcijo vseh značilk na voljo. Kot vhod dodamo še 15 minutne podatke o temperaturi za isto obdobje. Ponovno izpišemo izračunane značilke in značilke, ki niso na voljo.

- ex3.py -- ponovno ekstrakcija vseh značilk na voljo. Tokrat za podatke z granulacijo enega dne. Več značilk ni na voljo, ker za mnoge značilke npr. povprečno razmerje popoldanske in dopoldanske porabe potrebujemo najmanj granulacijo ene ure (obdobja dneva so definirana z urami).

## Primer dodajanje nove značilke v featureExtractor:
Dodali bomo značilko 'c_ht_var' - varianco porabe v visokotarifnih obdobjih. Potrebovali bomo podatke o tem kdaj so visokotarifna obdobja in značilko 'c_ht', ki je povprečna poraba v visokotarifnih obdobjih. Poleg tega bomo potrebovali granulacijo podatkov vsaj 60 minut (ht obdobja so definirana prek ure natančno). Na konec razreda featureExtractor dodamo (brez številk vrstic):

	 @_min_granularity(60)
	@_import_data(['hts', 'consumption'])
	@_import_features(['c_ht'])
	 @_check_if_exists_and_save_feature
	 def  c_ht_var(self):
			""" variance of consumption during hts. """
			return ((consumption[hts] - c_ht)**2).sum()/len(consumption[hts])


1. @_min_granularity(60) # pove razredu, da je ta značilka na voljo, samo v primeru dovolj granuliranih podatkov
2. @\_needs_data('hts') # iz slovarja extractor.\_data uvozimo ze prej izracunane podatke o tem kdaj so visokotarifna obdobja. V definiciji funkcije je nato 'hts' na voljo kot lokalna spremenljivka (hts je seznam true/false vrednosti enake dolzine kot podatki o porabi. Za vsak vnos v podatkih o porabi pove ali je v obdobju viskoih tarif ali ne)
3. @_needs_features('c_ht') # pove ekstraktorju naj najprej izracuna 'c_ht' in jo uvozi kot lokalno spremenljivko
4. @_check_if_exists_and_save #v extractor.extracted preveri, ce smo ze prej izracunali znacilko 'c_ht_var', da ne bomo po nepotrebnem racunali se enkrat. V nasprotnem primeru, po izvedeni funkciji, vrnjeno vrednost shrani v slovar extractor.extracted s kljucem 'c_ht_var'  (za kljuc vzame ime spodaj definirane funkcije)



