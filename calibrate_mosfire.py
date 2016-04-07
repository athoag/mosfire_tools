#! /usr/bin/python

import numpy as np
import os, sys
import glob
import pyfits as p
import matplotlib.pyplot as plt
import pywcs
import calibrate1Dspectrum as cal
from scipy import interpolate
import extract1D as ex1D
from astropy import modeling
from tools_willdawson import ra2deg, dec2deg
from scipy import ndimage, integrate
import gaussian_slit_loss as gsl
from check_mosfire import diagnostic_plots


#=======================================================================
#+
# NAME:
#   calibrate_mosfire.py
#
# PURPOSE:
#   Use the telluric standard from 2013 Dec 15 to calibrate MOSFIRE fluxes from any run
#
#	
# 
# BUGS:
# 
#
# Needed Fixes:
# 		
# 	
# 		
# REVISION HISTORY:
#   2014-05-16  started by Hoag (UC Davis)
#
#-
#=======================================================================

Ytp = '/Users/Austin/progs/XTcalc_dir/mosfire/Y_tp_tot.txt'

# def filter_curves():	
# 	hst_f105w = '/Users/Austin/progs/lephare_dev/Filt/hst/wfc3_f105w.pb' 
# 	hstx, hsty = np.genfromtxt(hst_f105w,usecols=(0,1),unpack=True)
# 	plt.plot(mosx,mosy,color='b')
# 	plt.plot(hstx,hsty,color='r')

def interp_wave(origwave,origspec,newwave):
		"""
	    ---- PURPOSE ----
	    Calculate interpolated spectrum of an object over a different wavelength array 

	    ---- INPUT ----
	    origwave        The original wavelength array [A]
	    origspec		The spectrum whose corresponding wavelength vector is origwave
	    newwave         The desired wavelength array [A] to which you want to interpolate origspec 

	    ---- OUTPUT ----
	    interp_spec		The desired spectrum whose corresponding wavelength array is newwave

	    """
		interp_foo = interpolate.interp1d(origwave,origspec)
		interp_spec = interp_foo(newwave)
		return interp_spec

class calibrator(object):
	''' Calibrate MOSFIRE Y-band observations. 
	WARNING!! WILL NOT WORK FOR DATA OTHER THAN Y-BAND!!!! '''
	def __init__(self):
		self.datemos='2013/12/14' 
		self.teleps1D_dec15='/Users/Austin/MOSFIRE_dec2013/REDUX/LONGSLIT-3x0.7_Y/LONGSLIT-3x0.7_Y_crop_eps_center179_aper5_1Dspec.fits' 
		self.teleps1D = self.teleps1D_dec15
	
	def scale_model_star(self,plot=False):
		''' Needed files/variables '''
		cstar = '/Users/Austin/science/MOSFIRE/MOSFIRE3_dec2013/Model_Calibration_Star/alpha_lyr_stis_007.fits'
		magV_tel = 9.05 # looked up on SIMBAD using coordinates 23:16:38.299 -19:57:42.50 -> object is: HD 219545 
		radecrcs = ['00:14:22.043','-30:23:14.25'] # Taken near the cluster core. Needed for correcting telluric
		telfirst = '/Users/Austin/MOSFIRE_dec2013/DATA/2013dec15/m131215_0117.fits'
		tellast = '/Users/Austin/MOSFIRE_dec2013/DATA/2013dec15/m131215_0158.fits'
		telfirstfits = p.open(telfirst)
		tellastfits = p.open(tellast)
		starttime = ':'.join(telfirstfits[0].header['TIME-OBS'].split(':')[0:2])
		endtime = ':'.join(tellastfits[0].header['TIME-END'].split(':')[0:2])
		teldate = telfirstfits[0].header['DATE-OBS']
		telstart = teldate + ' ' + starttime
		telend = teldate + ' ' + endtime
		extvalcal = 0.0297
		mosYtp = '/Users/Austin/progs/XTcalc_dir/mosfire/Y_tp_tot.txt'
		mosYtp_x = np.loadtxt(mosYtp,usecols=(0,))*10000  # scaled to get into Angstroms
		mosYtp_y = np.loadtxt(mosYtp,usecols=(1,))  
		################
		''' Load model star fits table and separate into wavelength and flux arrays '''
		cstarfits = p.open(cstar)
		cwaves = [cstarfits[1].data[i][0] for i in range(len(cstarfits[1].data))]
		cfluxes = [cstarfits[1].data[i][1] for i in range(len(cstarfits[1].data))]
		''' Read in telluric '''
		telwave, telspec_1D, tel_radec = cal.readfitsspec1D(self.teleps1D,spec='SPEC1D_SUM')
		goodmask = np.where(telspec_1D!=0)[0]
		# print goodmask
		goodwaves = telwave[goodmask]
		good_telspec1D = telspec_1D[goodmask] # in e/s/pix
		''' Interpolate standard star spectrum over telluric wavelengths'''
		spec_stan_interp = interp_wave(cwaves,cfluxes,goodwaves) 
		''' Interpolate throughput wavelength axis to match telluric's wavelength vector '''
		interptp = interp_wave(mosYtp_x,mosYtp_y,goodwaves) 
		''' rescale calibration star spectrum to match telluric's magnitude '''
		magABstan = cal.magFromSpec(goodwaves,spec_stan_interp,interptp)  # OK to do for standard star because its flux table is in erg/s/cm2/A. NOT OK to do for telluric.
		# print magABstan
		magABtel = magV_tel+magABstan
		# print "standard star magnitude = ", magABstan
		# print "Telluric magnitude = ", magABtel
		spec_stanscale = cal.scalespec(goodwaves,spec_stan_interp,interptp,magABtel) # scaling standard star spectrum to the apparent magnitude of the telluric
		''' Correct telluric for airmass '''	
		spec1D_tel_AMCOR = cal.correct_airmass(goodwaves,good_telspec1D,radecrcs,self.datemos,telstart,telend,plot=0)
		''' Correct telluric for extinction '''
		spec1D_tel_EXTCOR = cal.correct_galacticext(goodwaves,spec1D_tel_AMCOR,extvalcal,extlaw='Cardelli',plot=0)
		''' Correct for telluric absorption and telescope sensitivity '''
		# telcorrection = cal.correct_telluric(goodwaves,goodspec,c_corr)  # correction factors [erg/s/cm2/A] / [e/s/pix]
		telcorrection = cal.correct_telluric(goodwaves,spec1D_tel_EXTCOR,spec_stanscale)  # correction factors [erg/s/cm2/A] / [e/s/pix]
		if plot:
			plt.plot(goodwaves,np.log10(good_telspec1D),label='telluric')
			plt.plot(goodwaves,np.log10(spec1D_tel_AMCOR),label='airmass corrected')
			plt.plot(goodwaves,np.log10(spec1D_tel_EXTCOR),label='airmass+ext corrected')
			plt.plot(goodwaves,np.log10(spec_stanscale),label='scaled standard')
			plt.legend()
		return goodwaves,telcorrection

class spectrum(object):	
	''' Extract flux-calibrated 1D spectra from Y-band '''
	def __init__(self,cluster,scidir,ID,hst_date,utc_date,datadir,firstframe,lastframe,maskdir):
		''' 
		-----INPUT-------
		cluster 		'MACS0744', e.g.
		scidir 			directory to eps files
		ID 				'7302' e.g.
		hst_date 		The HST date when observations started, e.g. '2016/02/22'
		utc_date 		The UTC date when observations started, e.g. '2016/02/23'
		datadir 		path to data files of the 'm060216_0034.fits' king
		firstframe 		Integer for starting frame for observations during which the mask containing ID was targeted
		lastframe 		Integer for ending frame for observations during which the mask containing ID was targeted 
		maskdir 		The directory in which the MAGMA output mask was created
		'''
		self.cluster = cluster
		self.scidir = scidir
		self.ID = ID
		self.hst_date = hst_date
		self.utc_date = utc_date
		self.hyphen_date = self.utc_date.replace('/','-')
		self.datadir = datadir
		self.datestr = self.find_datestr()
		self.firstframe = firstframe
		self.lastframe = lastframe
		self.firstfits = self.findfits(self.firstframe)
		self.lastfits = self.findfits(self.lastframe)
		self.starttime = self.hyphen_date + ' ' +  ':'.join(p.getheader(self.firstfits)['UTC'].split(':')[0:2])
		self.endtime = self.hyphen_date + ' ' +  ':'.join(p.getheader(self.lastfits)['UTC'].split(':')[0:2])
		self.maskdir = maskdir
		cal_obj = calibrator()
		self.good_telwaves, self.telcorrection = cal_obj.scale_model_star()
		self.teleps1D = cal_obj.teleps1D
		self.epsfile, self.snrsfile, self.varfile = self.identify_files()
		self.uncal_spectrum = ex1D.extractor(ID=self.ID,scidir=self.scidir,date=self.utc_date)
		self.radec = self.get_radec()
		self.extvalcluster = self.get_cluster_extinction()
		self.photfits = '/Users/Austin/data/%s/photometry/Kuang/hst_%s_clash_psfmatch_60mas.fits' % (self.cluster,self.cluster)
		self.pixscale = 1.087 # Angstroms per pixel in the wavelength direction for MOSFIRE
		self.mosband = 'Y'


	def find_datestr(self):
		''' 
		-----PURPOSE-----
		Find the date string that is in the filename of all of the m*fits files
		-----INPUT-------
		'''
		# datestr = glob.glob('%s/m*fits' % self.datadir)[0].split('_')[0].split('m')[-1]
		datestr = glob.glob('%s/m*fits' % self.datadir)[0].split('/')[-1].split('_')[0].split('m')[-1]
		return datestr

	def findfits(self,frameno):
		''' 
		-----PURPOSE-----
		Find the filename from a frame number string that is in the filename of all of the m*fits files
		-----INPUT-------
		'''
		full_id = '0'*(4-len(str(frameno))) + '%s' % str(frameno)
		framename = self.datadir + '/m' + self.datestr + '_' + full_id + '.fits'
		return framename

	def identify_files(self):
		''' 
		-----PURPOSE-----
		Find the eps, snrs, and sig (variance) files based on the ID
		-----INPUT-------
		'''
		objfiles = glob.glob('%s/*%s*' % (self.scidir,self.ID) )
		epsfiles = [x for x in objfiles if 'eps' in x]
		snrsfiles = [x for x in objfiles if 'snrs' in x]
		varfiles = [x for x in objfiles if 'sig' in x]
		assert len(epsfiles) == 1
		assert len(snrsfiles) <= 1
		assert len(varfiles) <= 1
		epsfile = epsfiles[0]
		if snrsfiles != []:
			snrsfile = snrsfiles[0]
		else:
			snrsfile = None
		if varfiles != []:
			varfile = varfiles[0]
		else:
			varfile = None
		return epsfile, snrsfile, varfile

	def get_radec(self):
		''' 
		-----PURPOSE-----
		return the ra and dec in a list of strings to be read by the various 
		calibration methods in Kasper's calibrate1Dspectrum.py script
		-----INPUT-------
		'''
		maskname = self.maskdir.split('/')[-1]
		coordsfile = self.maskdir + '/%s.coords' % maskname
		lines = open(coordsfile).readlines()
		goodline = [line for line in lines if self.ID in line.split()[0]]
		assert len(goodline) == 1
		linestring = goodline[0]
		ra_sex = ':'.join(linestring.split()[3:6])
		dec_sex = ':'.join(linestring.split()[6:9])
		ra_deg = ra2deg(ra=ra_sex)
		dec_deg = dec2deg(dec=dec_sex)
		# ra = str(p.getheader(self.epsfile)['RA'])
		# dec = str(p.getheader(self.epsfile)['DEC'])
		radec = [ra_deg,dec_deg]
		return radec

	def get_ABmag_HST(self,band='f105w'):
		''' 
		-----PURPOSE-----
		Calculate the AB magnitude from the HST photometry in the nearest band to MOSFIRE Y-band
		-----INPUT-------
		band 		The photometric band in which you want magnitudes, default is 'f105w'
		'''
		ra, dec = self.radec
		# mags = p.open(fitscat)[1].data['%s_mag_iso' % band]
		mags = p.open(self.photfits)[1].data['%s_mag_AUTO' % band]
		phot_ras = p.open(self.photfits)[1].data['ALPHA_J2000']
		phot_decs = p.open(self.photfits)[1].data['DELTA_J2000']
		thresh=0.35
		mindist=thresh
		for jj in range(len(mags)):
			phot_ra, phot_dec = phot_ras[jj], phot_decs[jj]
			mag = mags[jj]
			# phot_F125W_magerr = phot_F125W_magerrs[jj]
			dist = np.sqrt(((ra-phot_ra)*3600*np.cos(np.pi/180*phot_dec))**2 + ((dec-phot_dec)*3600)**2) # in arcseconds
			if dist < mindist:
				keep_mag = mag
				keep_ra = phot_ra
				keep_dec = phot_dec
				# keep_id = phot_id
				mindist=dist
		if mindist >= thresh: # no match
			sys.exit("NO MATCH")
		return keep_mag	

	def get_cluster_extinction(self):
		''' 
		-----PURPOSE-----
		Get the extinction of the cluster from the lookup table
		-----INPUT-------
		'''
		ebmv_table = '/Users/Austin/observing/ref/ebmv_table.txt'
		clusters, ebmvs = np.genfromtxt(ebmv_table,unpack=True,usecols=(0,1),dtype='S20')
		ebmvs = map(float,ebmvs)
		ebmv_dict = {m:n for m,n in zip(clusters,ebmvs)}
		ebmv_cluster = ebmv_dict[self.cluster]
		return ebmv_cluster
	
	def get_seeing(self):
		''' 
		-----PURPOSE-----
		Get the median FWHM of the seeing from observations of a star on the mask

		WARNING!!! THE CENTER_TRACES OPTION IS HARDCODED FOR FEB 23, 2016 OBSERVATIONS
		-----INPUT-------
		-----OUTPUT------
		median_seeing_arcsec
		'''
		assert self.datestr == '160223', "Center traces for seeing calculation from star on mask are hardcoded to Feb 23, 2016 Run"
		seer = diagnostic_plots(numfiles=[self.firstframe,self.lastframe],center_traces=[1377, 1391],x_range=[600,1680],exclude=[])
		norm_fluxes, A_delta_ys, B_delta_ys, seeing_fwhms = seer.track_all(plot=False)
		median_seeing = np.median(seeing_fwhms)
		median_seeing_arcsec = median_seeing*0.18
		return median_seeing_arcsec
	
	def extract_calibrated_spectrum(self,spectype):
		''' 
		-----PURPOSE-----
		Extract the calibrated spectrum in erg/s/cm^2/AA. This is not normalized to HST yet, so that is a final step 
		that needs to take place for science spectra.
		-----INPUT-------
		spectype 		Any of the column names in the multi-dimensional fits table, e.g. 'SPEC1D_SUM', 'NOISE'
		-----OUTPUT------
		wave_wheretel		Wavelength vector in Angstroms where the spectrum is calibrated
		spec1D_FCAL			Flux-calibrated (flux normalization NOT considered) science spectrum
		'''
		spec1D_uncal = self.uncal_spectrum.extract_spectrum(spectype=spectype,aperwidth=5) # fixed aperwidth at the same used to extract standard star
		wave = self.uncal_spectrum.extract_spectrum(spectype='WAVELENGTH',aperwidth=5)
		# wave_goodwave, spec1D_uncal_goodwave = wave[goodwaves], spec1D_uncal[goodwaves]
		telcorrpix     = np.where((wave >= min(self.good_telwaves)) & (wave <= max(self.good_telwaves)))[0]
		wave_wheretel       = wave[telcorrpix]
		spec1D_wheretel = spec1D_uncal[telcorrpix]
		''' Correct for airmass '''
		spec1D_AMCOR  = cal.correct_airmass(wave_wheretel,spec1D_wheretel,self.radec,self.hst_date,self.starttime,self.endtime,plot=0)
		# spec1D_AMCOR  = cal.correct_airmass(wave_wheretel,spec1D_wheretel,self.radec,self.starttime,self.endtime,plot=0)
		''' Correct for extinction '''
		spec1D_EXTCOR = cal.correct_galacticext(wave_wheretel,spec1D_AMCOR,self.extvalcluster,extlaw='Cardelli')
		''' correct for telluric absorption and telescope losses '''
		spec1D_TELCOR = spec1D_EXTCOR * self.telcorrection

		spec1D_FCAL  = spec1D_TELCOR 

		return wave_wheretel, spec1D_FCAL



class bright_object(spectrum):

	def __init__(self,cluster,scidir,ID,hst_date,utc_date,datadir,firstframe,lastframe,maskdir):
		spectrum.__init__(self,cluster,scidir,ID,hst_date,utc_date,datadir,firstframe,lastframe,maskdir)

	def get_ABmag_MOSFIRE(self):
		''' 
		-----PURPOSE-----
		Calculate the AB magnitude from the MOSFIRE spectrum
		-----INPUT-------
		'''
		wave_cal, spec_cal = self.extract_calibrated_spectrum(spectype='SPEC1D_SUM')
		# Interpolate throughput wavelength axis to match brightwave
		TPx = np.loadtxt(Ytp,usecols=(0,))*10000  # scaled to get into Angstroms
		TPy = np.loadtxt(Ytp,usecols=(1,))  
		interptpspec = interp_wave(TPx,TPy,wave_cal) 
		magAB = cal.magFromSpec(wave_cal,spec_cal,interptpspec)
		return magAB

	def flux_normalization(self):
		hstmag = self.get_ABmag_HST(band='f105w')
		mosmag = self.get_ABmag_MOSFIRE() 
		magnorm = float(hstmag/mosmag)  # what you multiply magnitude extracted from mosfire by to get magnitude to compare to HST magnitude.
		fluxnorm = 10**((hstmag-mosmag)/(-2.5)) # what you multiply flux density (F_lambda) extracted mosfire by to get flux density to compare to HST flux density 
		return magnorm,fluxnorm

class science_object(spectrum):
	''' Extract flux-calibrated 1D spectra of a faint science object.
	Could speed this up by using a lookup table for various things such as AB magnitude rather than doing
	the crossmatch each time I load the class ''' 
	
	def __init__(self,cluster,scidir,ID_sci,ID_bright,hst_date,utc_date,datadir,firstframe,lastframe,maskdir):
		spectrum.__init__(self,cluster,scidir,ID_sci,hst_date,utc_date,datadir,firstframe,lastframe,maskdir)
		self.ID_bright = ID_bright
		self.photfits = '/Users/Austin/data/%s/photometry/Kuang/hst_%s_clash_psfmatch_60mas.fits' % (self.cluster,self.cluster)
		spectrum_bright = bright_object(cluster,scidir,self.ID_bright,hst_date,utc_date,datadir,firstframe,lastframe,maskdir,)
		self.magnorm, self.fluxnorm = spectrum_bright.flux_normalization()
		self.ABmag_HST = self.get_ABmag_HST(band='f105w')

	def extract_normalized_spectrum(self,spectype):
		''' 
		-----PURPOSE-----
		Extract the calibrated science spectrum of type 'spectype'
		-----INPUT-------
		spectype 		Any of the column names in the multi-dimensional fits table, e.g. 'SPEC1D_SUM', 'NOISE'		
		-----OUTPUT------
		wave_cal		Wavelength vector in Angstroms where the spectrum is calibrated
		spec1D_cal		Flux-calibrated (flux normalization IS considered) science spectrum
		'''
		wave, spec1D_nofluxnorm = self.extract_calibrated_spectrum(spectype=spectype)
		spec1D_calibrated = spec1D_nofluxnorm*self.fluxnorm
		return wave, spec1D_calibrated

	def plot_spectrum(self,spectype,smooth=False):
		''' 
		-----PURPOSE-----
		Plot the 1D flux calibrated science spectrum of type 'spectype'
		-----INPUT-------
		spectype 		Any of the column names in the multi-dimensional fits table, e.g. 'SPEC1D_SUM', 'NOISE'		
		smooth 			If True, will smooth the spectrum to the resolution of the instrument
		-----OUTPUT------
		'''
		wave,spec1D_calibrated = self.extract_normalized_spectrum(spectype=spectype)
		if smooth:
			smoothed_spec = ndimage.gaussian_filter1d(np.float_(spec1D_calibrated),sigma=3) # sigma in units of binsize of first argument.
			spec1D = smoothed_spec
		else:
			spec1D = spec1D_calibrated
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(wave,spec1D)
		ax.set_xlabel(r"$\lambda/\AA$",fontsize=18)
		ax.set_ylabel(r"%s [$\mathrm{erg\,s^{-1} cm^{-2}} \AA^{-1}$]" % spectype,fontsize=18)
		plt.tight_layout()

	def fit_gaussian(self,EL_wave,wave_range,plot=False):
		''' 
		-----PURPOSE-----
		Fit a 1D gaussian to an emission line flux density
		-----INPUT-------
		EL_wave 		Cental wavelength of the emission line [Angstroms]
		wave_range	    [min,max] where min and max are the wavelenghts in Angstroms
						that start and end the range over which you want to fit 
		plot 			if True, will plot the flux density and the 1D Gaussian fit 
		-----OUTPUT------
		'''
		# wave,spec1D_calibrated = self.extract_normalized_spectrum(spectype='SPEC1D_SUM')
		wave,spec1D_calibrated = self.extract_normalized_spectrum(spectype='SPEC1D_NORMIVAR')
		line_mask = np.logical_and(wave >= min(wave_range),wave <= max(wave_range))
		waves_line = wave[line_mask]
		spec1D_line = spec1D_calibrated[line_mask]
		g_init = modeling.models.Gaussian1D(amplitude=max(spec1D_line), mean=EL_wave, stddev=3) # initialize with amplitude equal to the maximum flux density value, mean equal to the line center and standard deviation equal to the spectral resolution of MOSFIRE
		fit_g = modeling.fitting.LevMarLSQFitter()
		g = fit_g(g_init, waves_line, spec1D_line)
		if plot:
			plt.step(waves_line,spec1D_line,color='r',label='flux density')
			plt.plot(waves_line,g(waves_line),color='b',label='Gaussian fit')
			plt.legend()
		return waves_line, g	
	
	def line_flux(self,EL_wave,wave_range=None,plot=False):
		''' 
		-----PURPOSE-----
		Extract the flux from the emission line from the 'SPEC1D_NORMIVAR' flux density
		-----INPUT-------
		EL_wave 		Cental wavelength of the emission line [Angstroms]
		
		wave_range	    [min,max] where min and max are the wavelenghts in Angstroms
						that start and end the range over which you want to fit. 
						By default (None) will use [EL_wave-25, EL_wave+25]
		
		plot 			if True, will plot the flux density and the 1D Gaussian fit 
		-----OUTPUT------
		line_flux 		In erg/cm^2/s
		'''
		if wave_range == None:
			wave_range = [EL_wave-25,EL_wave+25]
		waves_line, g = self.fit_gaussian(EL_wave=EL_wave,wave_range=wave_range)
		line_flux = sum(g(waves_line))
		# print "Line flux is %.4g erg/s/cm^2" % line_flux
		wave,spec1D_calibrated = self.extract_normalized_spectrum(spectype='SPEC1D_NORMIVAR')
		line_mask = np.logical_and(wave >= min(wave_range),wave <= max(wave_range))
		waves_line = wave[line_mask]
		spec1D_line = spec1D_calibrated[line_mask]
		if plot:
			plt.step(waves_line,spec1D_line,color='r',label='flux density')
			plt.plot(waves_line,g(waves_line),color='b',label='Gaussian fit')
			plt.legend()	
		return line_flux

	def lya_luminosity(self,EL_wave):
		''' 
		-----PURPOSE-----
		Calculate the Lyman-alpha luminosity of the emission line from its line flux
		-----INPUT-------
		EL_wave 		Cental wavelength of the emission line [Angstroms]
		wave_range	    [min,max] where min and max are the wavelenghts in Angstroms
						that start and end the range over which you want to fit 
		-----OUTPUT------

		'''
		from cosmolopy import cd
		cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.7}
		line_flux = self.line_flux(EL_wave=EL_wave)
		z_lya = EL_wave/1215.67 - 1
		D_M_Mpc = cd.comoving_distance_transverse(z_lya,**cosmo)
		cm_per_Mpc = 3.08567758E24
		D_M_cm = D_M_Mpc * cm_per_Mpc
		D_L_cm = (1+z_lya) * D_M_cm
		L_lya = 4*np.pi*D_L_cm**2*line_flux
		print "Lyman-alpha Luminosity is: %.4g erg/s" % L_lya
		return L_lya
	
	def lya_EW(self,EL_wave,wave_range=None,plot=False):
		''' 
		-----PURPOSE-----
		Calculate the rest-frame equivalent width (EW) of an emission line assuming it is Lyman-alpha
		by fitting the inverse-variance-weighted flux density to a 1D Gaussian using self.fit_gaussian()
		-----INPUT-------
		EL_wave 		Cental wavelength of the emission line [Angstroms]
		wave_range	    [min,max] where min and max are the wavelenghts in Angstroms
						that start and end the range over which you want to fit 
						By default (None) will use [EL_wave-25, EL_wave+25]
		plot 			if True, will plot the spectrum and filled in area with EW
		-----OUTPUT------
		'''
		if wave_range == None:
			wave_range = [EL_wave-25,EL_wave+25]
		waves_line, g = self.fit_gaussian(EL_wave=EL_wave,wave_range=wave_range)
		f_nu_HST = 10**(-1/(2.5)*(self.ABmag_HST+48.6)) # this is the flat flux density from HST in frequency units, f_nu
		'''convert to f_lambda, the useful quantity for calculating EW.
		There is a short explanation of how to do this here: https://en.wikipedia.org/wiki/AB_magnitude#Expression_in_terms_of_f.CE.BB
		Take the standard convention where f_nu = lamba^2 / c^2 * f_lambda.
		To make the expression exact over a particular bandpass, using the pivot wavelength, which can be looked up here:
		http://www.stsci.edu/hst/wfc3/analysis/ir_phot_zpt'''
		lambda_pivot = 10552. # pivot wavelength in A
		cval = 2.99792458e18 # speed of light in A/s
		# dlambda = 1570*2 # wavelength difference in A
		f_lambda_HST = cval*f_nu_HST/lambda_pivot**2
		# print "f_lambda from photometry is %.4g" % f_lambda_HST
		EW_observed = -1*integrate.trapz(1-g(waves_line)/f_lambda_HST,x=waves_line) # factor of -1 out front because of the way is usually defined (for absorption lines)
		# print "Equivalent width (observed) = %.2f " % EW_observed
		z_lya = EL_wave/1215.67 - 1
		EW_rest = EW_observed / (1+z_lya)
		print "The (rest-frame) Lyman-alpha EW is %.2f" % EW_rest
		if plot:
			wave,spec1D_calibrated = self.extract_normalized_spectrum(spectype='SPEC1D_NORMIVAR')
			line_mask = np.logical_and(wave >= min(wave_range),wave <= max(wave_range))
			spec1D_line = spec1D_calibrated[line_mask]
			plt.step(waves_line,spec1D_line,color='r',label='MOSFIRE flux density')
			plt.plot(waves_line,g(waves_line),color='b',label='Gaussian fit')
			plt.plot(waves_line,[f_lambda_HST for x in waves_line],color='cyan',label='Continuum flux density')
			plt.legend()	
		return EW_rest

	def limit_spectrum(self,aperwidth=5,Nsigma=1,plot=False):
		''' 
		-----PURPOSE-----
		Calculate the rest-frame equivalent width (EW) of an emission line assuming it is Lyman-alpha
		by fitting the inverse-variance-weighted flux density to a 1D Gaussian using self.fit_gaussian()
		-----INPUT-------
		aperwidth 		vertical size of extraction aperture in pixels
		Nsigma 			The number of standard deviations at which you want to calculate the flux limit. Default is 1-sigma
		plot 			if True, will plot the spectrum and filled in area with EW
		-----OUTPUT------
		'''
		wave,spec1D_std = self.extract_normalized_spectrum(spectype='SPEC1DERR_STDSIGNAL')
		# print np.median(spec1D_std)
		fwhm_seeing = self.get_seeing() # in arcseconds
		print "Seeing FWHM was %.2f arcseconds" % fwhm_seeing
		slitloss = gsl.relative_loss(fwhm_seeing=fwhm_seeing)
		print "Relative slit loss was %.2f" % slitloss
		dlambda = 3 # dlambda [AA] should be ~ the spectral resolution of MOSFIRE, which is 3 Angstroms in Y-band. Treu et al. 2012 Figure 6 say for an "unresolved" line
		assert self.mosband == 'Y', "Dlambda may change for observations that are not Y-band"
		flux_limit = spec1D_std*np.sqrt(aperwidth*2*dlambda/self.pixscale)*Nsigma*self.pixscale/(1-slitloss) # dlambda/pixscale is FWHM in pixels, so 2*FWHM is the spectral dimension of the aperture 
		# wave, flux_noise = extract_noise(date=date,spec1D_sci=scifits) 
		# flux_ivar = np.divide(1.0,np.square(flux_noise))
		
		# return wave, flux_limit, flux_ivar