#!/usr/bin/env python3
"""Extrae propiedades, espectros y poblaciones de salidas STARLIGHT .out.

Uso: python -m src.data.starlight_extractor data/starlight --output data/processed/starlight

Todos los archivos de salida se nombran con la columna ``name`` del manifest
(``manga-10001-12701``, sin el sufijo ``-1D`` de los .out); ``indice.csv`` y los .json la
traen como ``name``. Además de los .npz/.json, escribe en ``<output>/espectros_fits`` un FITS
por galaxia (``<name>-1D.fits``, el nombre que busca el dataloader) con la misma estructura
que los de MaNGA (extensión ``Spectra`` con columnas ``WAVE`` y ``FLUX_FoV``), de modo que ``src.dataloader.spectra`` los consuma
sin cambios: ``--fits-dir <output>/espectros_fits --channels FLUX_FoV``.
STARLIGHT trabaja en reposo, mientras que el dataloader divide ``WAVE`` entre ``1+nsa_z``;
por eso ``WAVE`` se escribe en el marco observado usando el ``nsa_z`` del manifest.
Requiere numpy y astropy; Python >=3.9.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

KEYS = {
    'N_base': 'n_base', 'N_chains': 'n_chains', 'Nl_eff': 'n_lambda_eff',
    'chi2/Nl_eff': 'chi2_nl_eff', 'adev (%)': 'adev_pct',
    'sum-of-x (%)': 'sum_x_pct', 'Flux_tot (units of input spectrum!)': 'flux_tot',
    'Mini_tot (???)': 'mini_tot', 'Mcor_tot (???)': 'mcor_tot',
    'v0_min  (km/s)': 'v0_kms', 'vd_min  (km/s)': 'vd_kms',
    'AV_min  (mag)': 'av_mag', 'YAV_min (mag)': 'yav_mag',
    'fobs_norm (in input units)': 'fobs_norm',
    'S/N in S/N window': 'sn_window', 'S/N in norm. window': 'sn_norm',
    'S/N_err in S/N window': 'sn_err_window',
    'l_norm (A) - for base': 'lambda_norm_angstrom',
}

def number(s):
    return float(s.replace('D', 'E').replace('d', 'e'))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECTRA_EXTENSION = 'Spectra'
NO_DATA_WEIGHT = -2  # píxeles sin dato observado en el bloque Synthetic spectrum


def load_redshifts(manifest):
    """Devuelve {name: nsa_z} desde el manifest (columnas name y nsa_z)."""
    with Path(manifest).open(newline='', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        if not {'name', 'nsa_z'} <= set(reader.fieldnames or []):
            raise ValueError(f'{manifest} debe contener las columnas name y nsa_z')
        return {row['name']: float(row['nsa_z']) for row in reader if row['nsa_z'] not in ('', None)}


def galaxy_name(ident):
    """'manga-10001-12701-1D' -> 'manga-10001-12701' (columna name del manifest)."""
    return ident.removesuffix('-1D')


def write_fits_spectrum(path, spec, fobs_norm, redshift, channel='FLUX_FoV'):
    """Escribe el espectro modelado con el esquema FITS que espera src.dataloader.spectra.

    ``WAVE`` queda en el marco observado (reposo * (1+z)) y el flujo en las unidades del
    espectro de entrada (f_syn * fobs_norm). Los píxeles sin dato observado se marcan NaN
    para que la máscara de cobertura del dataloader no incluya el modelo extrapolado.
    """
    if not np.isfinite(redshift) or redshift <= -1:
        raise ValueError(f'Redshift inválido: {redshift}')
    if not np.isfinite(fobs_norm) or fobs_norm == 0:
        raise ValueError(f'fobs_norm inválido: {fobs_norm}')
    observed = spec[:, 3] != NO_DATA_WEIGHT
    model = np.where(observed, spec[:, 2] * fobs_norm, np.nan)
    obs = np.where(observed, spec[:, 1] * fobs_norm, np.nan)
    suffix = channel.removeprefix('FLUX_')
    columns = [
        fits.Column(name='WAVE', format='D', array=spec[:, 0] * (1.0 + redshift)),
        fits.Column(name=channel, format='D', array=model),
        fits.Column(name=f'{channel}_OBS', format='D', array=obs),
        fits.Column(name=f'WEIGHT_{suffix}', format='D', array=spec[:, 3]),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name=SPECTRA_EXTENSION)
    hdu.header['Z'] = (redshift, 'nsa_z usado para pasar de reposo a observado')
    hdu.header['FOBSNORM'] = (fobs_norm, 'f_syn y f_obs escalados por este valor')
    hdu.header['SPECKIND'] = ('STARLIGHT best-fit model', f'{channel} = modelo, no observado')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def parse(path):
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    props = {'name': galaxy_name(path.name.removesuffix('.out')), 'source_file': path.name,
             'flux_units': 'normalized_to_fobs_norm',
             'mass_units_verified': False}
    # Read only header + best model; exclude subsequent chain summaries.
    best_end = next((i for i,l in enumerate(lines) if 'Synthesis Results - Average & Chains' in l), len(lines))
    for line in lines[:best_end]:
        match = re.match(r'^\s*([^\s#]+)\s+\[([^\]]+)\]', line)
        if match:
            raw, key = match.groups()
            key = key.strip()
            if key in KEYS:
                props[KEYS[key]] = number(raw)
    # Preserve S/N explicitly because comments are not always unique elsewhere.
    header = next((i for i,l in enumerate(lines) if l.startswith('# j') and 'x_j(%)' in l), None)
    if header is None:
        raise ValueError('No se encontró la tabla de poblaciones del mejor modelo')
    n = int(props.get('n_base', 0))
    if n != 45:
        raise ValueError(f'N_base={n}; se esperaba 45 para la matriz 2x45')
    pop_rows=[]
    for line in lines[header+1:header+1+n]:
        tokens=line.split()
        if len(tokens)<6 or not tokens[0].isdigit():
            raise ValueError('Fila de población incompleta')
        pop_rows.append((int(tokens[0]), *map(number,tokens[1:6])))
    pop=np.asarray(pop_rows, dtype=float)
    if not np.array_equal(pop[:,0], np.arange(1,n+1)):
        raise ValueError('Índices de población inesperados')
    if not np.all(np.isfinite(pop)):
        raise ValueError('Valores no finitos en poblaciones')
    matrix=np.stack([pop[:,1],pop[:,3]],axis=0).astype(np.float32) # luz y masa actual
    age=pop[:,4].astype(np.float64)
    metallicity=pop[:,5].astype(np.float64)
    props['x_sum_extracted_pct']=float(pop[:,1].sum())
    props['mcor_sum_extracted_pct']=float(pop[:,3].sum())
    props['mini_sum_extracted_pct']=float(pop[:,2].sum())
    props['age_mean_light_gyr']=float(np.dot(pop[:,1], age)/pop[:,1].sum()/1e9)
    props['age_mean_mass_gyr']=float(np.dot(pop[:,3], age)/pop[:,3].sum()/1e9)
    props['z_mean_light']=float(np.dot(pop[:,1], metallicity)/pop[:,1].sum())
    props['z_mean_mass']=float(np.dot(pop[:,3], metallicity)/pop[:,3].sum())
    props['light_young_lt_100myr_pct']=float(100*pop[age<1e8,1].sum()/pop[:,1].sum())
    props['mass_old_ge_1gyr_pct']=float(100*pop[age>=1e9,3].sum()/pop[:,3].sum())
    spec_hdr=next((i for i,l in enumerate(lines) if 'Synthetic spectrum (Best Model)' in l), None)
    if spec_hdr is None:
        raise ValueError('Falta el bloque Synthetic spectrum')
    spec_nmatch=re.match(r'^\s*(\d+)\s+\[Nl_obs\]', lines[spec_hdr+1])
    if not spec_nmatch:
        raise ValueError('Falta la longitud Nl_obs')
    spec_n=int(spec_nmatch.group(1))
    spec=np.array([[number(v) for v in l.split()[:4]] for l in lines[spec_hdr+2:spec_hdr+2+spec_n]],dtype=np.float64)
    if spec.shape!=(spec_n,4) or not np.all(np.isfinite(spec)):
        raise ValueError('Espectro incompleto o no finito')
    props['n_spectral_samples']=spec_n
    props['lambda_min_angstrom']=float(spec[0,0])
    props['lambda_max_angstrom']=float(spec[-1,0])
    return props, matrix, age, metallicity, spec

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path, help='Carpeta con archivos .out o archivo .out')
    ap.add_argument('--output', type=Path, default=Path('datos_starlight'))
    ap.add_argument('--manifest', type=Path, default=PROJECT_ROOT/'data/galaxy_manifest.csv',
                    help='CSV con name y nsa_z para reconstruir WAVE observado en los FITS')
    ap.add_argument('--channel', default='FLUX_FoV',
                    help='Columna de flujo del FITS (debe coincidir con la apertura de entrada de STARLIGHT)')
    ap.add_argument('--recursive',action='store_true',help='Buscar archivos .out en subcarpetas')
    args=ap.parse_args()
    files=([args.input] if args.input.is_file() else sorted(args.input.rglob('*.out') if args.recursive else args.input.glob('*.out')))
    if not files:
        ap.error('No se encontraron archivos .out')
    root=args.output
    for folder in ('propiedades','espectros_starlight','espectros_fits','poblaciones_estelares'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    redshifts=load_redshifts(args.manifest)
    entries=[]; failures=0
    reference=None
    for file in files:
        try:
            props, matrix, age, metallicity, spec=parse(file)
            if reference is None:
                reference=(age,metallicity)
            elif not (np.array_equal(age,reference[0]) and np.array_equal(metallicity,reference[1])):
                raise ValueError('La cuadrícula edad/Z no coincide con los otros archivos: no combinar sin armonizar')
            name=props['name']
            if name not in redshifts:
                raise ValueError(f'{name} no está en {args.manifest} (se requiere nsa_z)')
            props['nsa_z']=redshifts[name]
            write_fits_spectrum(root/'espectros_fits'/f'{name}-1D.fits', spec,
                                props['fobs_norm'], redshifts[name], args.channel)
            (root/'propiedades'/f'{name}.json').write_text(json.dumps(props,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            np.savez_compressed(root/'poblaciones_estelares'/f'{name}.npz',
                                matriz=matrix, edades_yr=age, metalicidades_z=metallicity,
                                canales=np.array(['luz_pct','masa_actual_pct']))
            np.savez_compressed(root/'espectros_starlight'/f'{name}.npz',
                                lambda_angstrom=spec[:,0], f_obs=spec[:,1],f_syn=spec[:,2],
                                residuo=spec[:,1]-spec[:,2], peso=spec[:,3],
                                mascara_valida=spec[:,3]>0)
            entries.append(props)
            print(f'OK {name}: poblaciones={matrix.shape}, espectro={len(spec)}')
        except Exception as exc:
            failures+=1
            print(f'ERROR {file.name}: {exc}',file=sys.stderr)
    if entries:
        with (root/'propiedades'/'indice.csv').open('w',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp,fieldnames=sorted(set().union(*(x.keys() for x in entries))))
            writer.writeheader();writer.writerows(entries)
        if reference is not None:
            np.savez_compressed(root/'poblaciones_estelares'/'base_poblaciones.npz',
                                edades_yr=reference[0],metalicidades_z=reference[1])
    print(f'Procesados: {len(entries)}; errores: {failures}. Salida: {root}')
    return 1 if failures else 0

if __name__=='__main__':
    sys.exit(main())
