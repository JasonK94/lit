import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import logging
from scipy import sparse
from typing import Optional, Dict, Any, Literal, List, Tuple # 타입 힌트 추가

# --- 로깅 설정 ---
# 파일과 콘솔에 모두 로깅하도록 설정
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocess_xenium')
logger.setLevel(logging.INFO) # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# 파일 핸들러 (함수 호출 시 동적으로 파일 경로 설정)
file_handler = None

# Scanpy 로깅 설정
sc.settings.verbosity = 3 # 오류, 경고, 정보 표시
sc.logging.print_header()
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False) # 그림 품질 향상 및 프레임 제거


def preprocess_xenium(
    # --- 입력 파일 경로 ---
    h5_url: str,
    cells_parquet_url: str,
    boundaries_parquet_url: str, # 현재 사용되지 않지만, 나중을 위해 포함
    core_info_dir: Optional[str] = None, # Core 정보 CSV 파일들이 있는 디렉토리 (선택 사항)

    # --- 출력 및 기본 설정 ---
    output_prefix: str = "processed_xenium", # 출력 파일명 접두사
    sample_id: str = "sample", # 샘플 ID 지정
    save_plots: bool = True, # 중간 과정 plot 저장 여부
    save_adata: bool = True, # 최종 AnnData 객체 저장 여부

    # --- QC 파라미터 ---
    qc_filter: bool = True, # QC 기반 필터링 수행 여부
    min_counts_per_cell: Optional[int] = 10, # 세포당 최소 count 수 (필터링용)
    max_counts_per_cell: Optional[int] = None, # 세포당 최대 count 수 (필터링용, 필요시 설정)
    min_genes_per_cell: Optional[int] = 3,   # 세포당 최소 유전자 수 (필터링용)
    max_genes_per_cell: Optional[int] = None, # 세포당 최대 유전자 수 (필터링용, 필요시 설정)
    min_cells_per_gene: Optional[int] = 3,   # 유전자당 최소 발현 세포 수 (필터링용)
    max_pct_mito: Optional[float] = 20.0, # 최대 미토콘드리아 유전자 비율 (필터링용)

    # --- Normalization & HVG 파라미터 ---
    normalization_method: Literal['lognorm', 'sct'] = 'lognorm', # 정규화 방법 선택
    # LogNorm 파라미터
    lognorm_target_sum: Optional[float] = 1e4, # normalize_total target_sum
    lognorm_hvg_min_mean: float = 0.0125,
    lognorm_hvg_max_mean: float = 3,
    lognorm_hvg_min_disp: float = 0.5,
    lognorm_hvg_n_top_genes: Optional[int] = None, # 상위 N개 HVG 선택 (선택 사항)
    # SCTransform 파라미터 (sctransform 라이브러리 필요)
    sct_n_genes: int = 3000, # SCTransform으로 찾을 HVG 수
    sct_min_cells: int = 5,    # SCTransform 유전자 필터링 기준
    sct_residual_type: Literal['pearson', 'deviance'] = 'pearson',

    # --- Scaling 파라미터 (lognorm 사용 시) ---
    scale_max_value: Optional[float] = 10, # sc.pp.scale의 max_value (clipping)

    # --- 차원 축소 및 클러스터링 파라미터 ---
    pca_n_comps: int = 50,
    neighbors_n_pcs: int = 40, # neighbors 계산에 사용할 PC 수 (elbow plot 참고)
    neighbors_n_neighbors: int = 15,
    cluster_algo: Literal['leiden', 'louvain'] = 'leiden',
    cluster_resolution: float = 0.8,
    umap_min_dist: float = 0.5,
    umap_spread: float = 1.0,

    # --- 마커 유전자 탐색 파라미터 ---
    rank_genes_method: Literal['wilcoxon', 't-test', 'logreg'] = 'wilcoxon',
    rank_genes_n_genes: int = 25, # plot할 마커 유전자 수
    rank_genes_pts: bool = True, # rank_genes_groups에서 pts 계산 여부

    # --- 기타 ---
    random_seed: int = 0 # 재현성을 위한 random seed
) -> Optional[sc.AnnData]:
    """
    Xenium 데이터셋을 로드하고 전처리하는 파이프라인 함수.

    Args:
        h5_url (str): cell_feature_matrix.h5 파일 경로.
        cells_parquet_url (str): cells.parquet 파일 경로.
        boundaries_parquet_url (str): cell_boundaries.parquet 파일 경로. (현재 사용 안 함)
        core_info_dir (Optional[str]): Core 정보 CSV 파일이 있는 디렉토리 경로. Defaults to None.
        output_prefix (str): 로그, 플롯, AnnData 파일 저장 시 사용할 접두사. Defaults to "processed_xenium".
        sample_id (str): 데이터셋에 부여할 샘플 ID. Defaults to "sample".
        save_plots (bool): 중간 분석 플롯 저장 여부. Defaults to True.
        save_adata (bool): 최종 AnnData 객체 저장 여부. Defaults to True.
        qc_filter (bool): QC 기반 세포/유전자 필터링 수행 여부. Defaults to True.
        min_counts_per_cell (Optional[int]): 필터링 기준: 세포당 최소 UMI 수. Defaults to 10.
        max_counts_per_cell (Optional[int]): 필터링 기준: 세포당 최대 UMI 수. Defaults to None.
        min_genes_per_cell (Optional[int]): 필터링 기준: 세포당 최소 발현 유전자 수. Defaults to 3.
        max_genes_per_cell (Optional[int]): 필터링 기준: 세포당 최대 발현 유전자 수. Defaults to None.
        min_cells_per_gene (Optional[int]): 필터링 기준: 유전자당 최소 발현 세포 수. Defaults to 3.
        max_pct_mito (Optional[float]): 필터링 기준: 최대 미토콘드리아 유전자 비율 (%). Defaults to 20.0.
        normalization_method (Literal['lognorm', 'sct']): 정규화 방법 ('lognorm' 또는 'sct'). Defaults to 'lognorm'.
        lognorm_target_sum (Optional[float]): LogNorm 시 normalize_total의 target_sum. Defaults to 1e4.
        lognorm_hvg_min_mean (float): LogNorm 시 HVG 기준: 최소 평균 발현. Defaults to 0.0125.
        lognorm_hvg_max_mean (float): LogNorm 시 HVG 기준: 최대 평균 발현. Defaults to 3.
        lognorm_hvg_min_disp (float): LogNorm 시 HVG 기준: 최소 분산. Defaults to 0.5.
        lognorm_hvg_n_top_genes (Optional[int]): LogNorm 시 상위 N개 HVG 선택 (다른 기준보다 우선). Defaults to None.
        sct_n_genes (int): SCTransform 시 찾을 HVG 수. Defaults to 3000.
        sct_min_cells (int): SCTransform 시 유전자가 최소 발현되어야 하는 세포 수. Defaults to 5.
        sct_residual_type (Literal['pearson', 'deviance']): SCTransform 잔차 타입. Defaults to 'pearson'.
        scale_max_value (Optional[float]): LogNorm 후 스케일링 시 최대값 clipping. Defaults to 10.
        pca_n_comps (int): PCA 주성분 개수. Defaults to 50.
        neighbors_n_pcs (int): Neighbors 계산 시 사용할 PC 개수. Defaults to 40.
        neighbors_n_neighbors (int): Neighbors 계산 시 이웃 수. Defaults to 15.
        cluster_algo (Literal['leiden', 'louvain']): 사용할 클러스터링 알고리즘. Defaults to 'leiden'.
        cluster_resolution (float): 클러스터링 해상도. Defaults to 0.8.
        umap_min_dist (float): UMAP min_dist 파라미터. Defaults to 0.5.
        umap_spread (float): UMAP spread 파라미터. Defaults to 1.0.
        rank_genes_method (Literal['wilcoxon', 't-test', 'logreg']): 마커 유전자 탐색 방법. Defaults to 'wilcoxon'.
        rank_genes_n_genes (int): 마커 유전자 플롯 시 상위 유전자 수. Defaults to 25.
        rank_genes_pts (bool): rank_genes_groups에서 pts 계산 여부. Defaults to True.
        random_seed (int): 재현성을 위한 random seed. Defaults to 0.

    Returns:
        Optional[sc.AnnData]: 전처리된 AnnData 객체. save_adata가 False면 None 반환 가능.
    """
    global file_handler # 전역 file_handler 사용 선언

    # --- 출력 디렉토리 생성 ---
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # --- 로깅 파일 핸들러 설정 ---
    # 기존 핸들러 제거 (함수 재호출 시 중복 방지)
    if file_handler:
        logger.removeHandler(file_handler)
    log_file = f"{output_prefix}_processing.log"
    file_handler = logging.FileHandler(log_file, mode='w') # 'w' 모드로 열어 매번 새로 작성
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    logger.info("Starting Xenium data preprocessing pipeline.")
    logger.info(f"Output prefix set to: {output_prefix}")
    logger.info(f"Sample ID set to: {sample_id}")

    # 재현성을 위한 random seed 설정
    np.random.seed(random_seed)
    
    
    # provenance 정보를 저장할 별도 딕셔너리 초기화
    provenance_data = {'pipeline_params': {}}
    # 함수 호출 시 사용된 파라미터 저장 (locals() 사용 시 주의 필요, 여기서는 주요 파라미터만 명시적으로 저장하는 것이 더 안전할 수 있음)
    # 예시: provenance_data['pipeline_params']['h5_url'] = h5_url ... 등
    # 또는 함수 시작 시 locals() 복사 후 불필요한 것 제거
    try:
        func_args = locals().copy()
        # 함수 내부에서만 사용되거나 저장할 필요 없는/직렬화 불가능한 객체 제거
        func_args.pop('adata', None)
        func_args.pop('adata_raw', None)
        func_args.pop('vst_out', None)
        func_args.pop('adata_hvg', None)
        func_args.pop('file_handler', None) # 전역 변수 제거
        func_args.pop('provenance_data', None) # 자기 자신 제거
        func_args.pop('conflict_report', None) # 리스트 객체 제거 (내용은 저장됨)
        # 필요시 더 많은 임시 변수 제거
        provenance_data['pipeline_params'] = func_args
    except Exception as e:
        logger.warning(f"Could not capture all pipeline parameters for provenance: {e}")
    # --- 1. 데이터 로딩 ---
    logger.info("--- Step 1: Loading Data ---")
    # provenance_data에 해당 단계 정보 기록 시작
    provenance_data['load'] = {'params': {}, 'log': {}}
    try:
        logger.info(f"Reading H5 file: {h5_url}")
        adata = sc.read_10x_h5(h5_url)
        adata.var_names_make_unique() # 유전자 이름 고유하게 만들기
        adata.obs['sample_id'] = sample_id # 샘플 ID 부여

        provenance_data['load']['params']['h5_url'] = h5_url
        provenance_data['load']['log']['initial_n_obs'] = adata.n_obs
        provenance_data['load']['log']['initial_n_vars'] = adata.n_vars
        provenance_data['load']['timestamp'] = pd.Timestamp.now().isoformat()
        adata.layers['counts'] = adata.X.copy() # 원본 count 저장

        logger.info(f"Reading cells parquet file: {cells_parquet_url}")
        cells_df = pd.read_parquet(cells_parquet_url)
        if 'cell_id' in cells_df.columns:
            cells_df = cells_df.set_index('cell_id')
        else:
            logger.warning("'cell_id' column not found in cells parquet. Assuming index matches AnnData.")

        # boundaries parquet는 현재 사용하지 않음 (필요시 로드)
        # logger.info(f"Reading boundaries parquet file: {boundaries_parquet_url}")
        # boundaries_df = pd.read_parquet(boundaries_parquet_url)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Aborting.")
        return None
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return None

    logger.info(f"Initial AnnData object shape: {adata.shape}")
    logger.info(f"Available obs columns from H5: {adata.obs.columns.tolist()}")
    logger.info(f"Available var columns from H5: {adata.var.columns.tolist()}")

    # --- 2. 메타데이터 통합 ---
    logger.info("--- Step 2: Integrating Metadata ---")
    provenance_data['metadata'] = {'params': {}, 'log': {}}

    # 2a. Cells Parquet 정보 추가
    original_obs_cols = set(adata.obs.columns)
    logger.info("Joining metadata from cells parquet file.")
    adata.obs = adata.obs.join(cells_df, how='left', lsuffix='_h5', rsuffix='_parquet')
    new_cols = set(adata.obs.columns) - original_obs_cols
    logger.info(f"Added columns from cells parquet: {list(new_cols)}")
    provenance_data['metadata']['log']['cells_parquet_cols_added'] = list(new_cols)

    # 2b. 공간 좌표 추가 (obsm['spatial'])
    # 우선순위: 'x_centroid', 'y_centroid' -> '_x', '_y' (Xenium v1)
    x_col, y_col = None, None
    if 'x_centroid' in adata.obs.columns and 'y_centroid' in adata.obs.columns:
        x_col, y_col = 'x_centroid', 'y_centroid'
    elif 'x' in adata.obs.columns and 'y' in adata.obs.columns: # Xenium v1 호환성
         x_col, y_col = 'x', 'y'
    # 필요시 fov 관련 좌표 추가 (보통 상대 좌표임)
    # elif 'fov_x' in adata.obs.columns and 'fov_y' in adata.obs.columns:
    #     x_col, y_col = 'fov_x', 'fov_y'

    if x_col and y_col:
        logger.info(f"Populating adata.obsm['spatial'] using columns: '{x_col}', '{y_col}'")
        adata.obsm['spatial'] = adata.obs[[x_col, y_col]].to_numpy()
        provenance_data['metadata']['log']['spatial_coords_source'] = (x_col, y_col)
    else:
        logger.warning("Could not find standard spatial coordinate columns (e.g., 'x_centroid', 'y_centroid', 'x', 'y') in adata.obs. adata.obsm['spatial'] not populated.")
        provenance_data['metadata']['log']['spatial_coords_source'] = None

    # 2c. Core 정보 추가 (선택 사항)
    conflict_report = []
    if core_info_dir:
        logger.info(f"Integrating core information from directory: {core_info_dir}")
        if not os.path.isdir(core_info_dir):
            logger.warning(f"Core info directory not found: {core_info_dir}. Skipping core info integration.")
        else:
            core_files = glob.glob(os.path.join(core_info_dir, '*_cells_stats.csv'))
            logger.info(f"Found {len(core_files)} potential core info files.")
            adata.obs['Core'] = pd.NA # Core 컬럼 초기화

            for core_url in core_files:
                basename = os.path.basename(core_url)
                # 파일명에서 숫자 Core ID 추출 (정규 표현식 강화)
                match = re.match(r"(\d+)[_-].*cells.*\.csv", basename, re.IGNORECASE)
                if not match:
                    logger.warning(f"Cannot extract numeric Core ID from filename: {basename}. Skipping file.")
                    continue
                core_id = match.group(1)
                logger.debug(f"Processing core file: {basename} for Core ID: {core_id}")

                try:
                    core_df = pd.read_csv(core_url, comment="#", skip_blank_lines=True)
                    # Cell ID 컬럼 이름 유연하게 찾기
                    cell_id_col = None
                    possible_id_cols = ['Cell ID', 'cell_id', 'Cell_ID']
                    for col in possible_id_cols:
                        if col in core_df.columns:
                            cell_id_col = col
                            break
                    if not cell_id_col:
                        logger.warning(f"'{possible_id_cols}' column not found in {basename}. Skipping.")
                        continue

                    core_df = core_df.set_index(cell_id_col)
                    core_df["Core_from_file"] = core_id # 임시 컬럼명 사용

                    # AnnData 인덱스와 Core 파일 인덱스 비교 (타입 통일 시도)
                    adata_idx = adata.obs.index.astype(str)
                    core_idx = core_df.index.astype(str)
                    overlapping_idx_str = adata_idx.intersection(core_idx)

                    if overlapping_idx_str.empty:
                        logger.warning(f"No overlapping cell IDs found between AnnData and {basename}. Check ID formats.")
                        continue

                    # 원래 AnnData 인덱스 타입으로 변환하여 사용
                    overlapping_idx = adata.obs.index[adata_idx.isin(overlapping_idx_str)]
                    core_df_aligned = core_df.loc[overlapping_idx_str] # 문자열 인덱스로 정렬된 core 데이터

                    logger.info(f"Found {len(overlapping_idx)} overlapping cells for Core {core_id} in {basename}.")

                    # Core ID 할당 (이미 할당된 경우 경고)
                    core_conflict_mask = adata.obs.loc[overlapping_idx, 'Core'].notna() & \
                                         (adata.obs.loc[overlapping_idx, 'Core'] != core_id)
                    if core_conflict_mask.any():
                         conflicted_cells = overlapping_idx[core_conflict_mask]
                         logger.warning(f"{len(conflicted_cells)} cells in Core {core_id} file ({basename}) were already assigned to a different Core: {conflicted_cells.tolist()[:5]}...")
                         # 충돌 시 어떻게 처리할지 정책 결정 필요 (예: 첫 번째 할당 유지, 파일 우선 등)
                         # 여기서는 일단 덮어쓰지 않음
                         adata.obs.loc[overlapping_idx[~core_conflict_mask], 'Core'] = core_id
                    else:
                         adata.obs.loc[overlapping_idx, 'Core'] = core_id

                    # 다른 컬럼들 병합 (충돌 처리 포함)
                    for col in core_df_aligned.columns:
                        if col == "Core_from_file": continue # 임시 컬럼 제외

                        col_target = f"{col}_core" # 이름 충돌 방지 위해 접미사 추가
                        if col_target not in adata.obs.columns:
                            adata.obs[col_target] = pd.NA # 새 컬럼 생성 (Pandas 1.x 이상 NA 사용 권장)

                        # Core 파일 값으로 업데이트 (기존 값이 NA인 경우만)
                        update_mask = adata.obs.loc[overlapping_idx, col_target].isna() & \
                                      core_df_aligned[col].notna().values # .values로 numpy 배열화
                        adata.obs.loc[overlapping_idx[update_mask], col_target] = core_df_aligned.loc[overlapping_idx_str[update_mask], col].values

                        # 충돌 기록 (기존 값과 새 값이 모두 있고 다른 경우)
                        conflict_mask = adata.obs.loc[overlapping_idx, col_target].notna() & \
                                        core_df_aligned[col].notna().values & \
                                        (adata.obs.loc[overlapping_idx, col_target].astype(str) != core_df_aligned.loc[overlapping_idx_str, col].astype(str)) # 타입 다를 수 있으므로 문자열 비교
                        if conflict_mask.any():
                            conflicted_cells_idx = overlapping_idx[conflict_mask]
                            conflicted_cells_str_idx = overlapping_idx_str[conflict_mask]
                            for i, cell_idx in enumerate(conflicted_cells_idx):
                                cell_str_idx = conflicted_cells_str_idx[i]
                                conflict_report.append({
                                    "cell": cell_idx,
                                    "column": col_target,
                                    "adata_value": adata.obs.loc[cell_idx, col_target],
                                    "core_value": core_df_aligned.loc[cell_str_idx, col],
                                    "source_file": basename
                                })
                            logger.warning(f"Value conflict for column '{col_target}' in {conflict_mask.sum()} cells from {basename}. See conflict report.")

                except pd.errors.EmptyDataError:
                    logger.warning(f"Core file is empty: {basename}. Skipping.")
                except Exception as e:
                    logger.error(f"Error processing core file {basename}: {e}")

            n_assigned = adata.obs['Core'].notna().sum()
            n_total = adata.n_obs
            logger.info(f"Assigned Core IDs to {n_assigned}/{n_total} cells.")
            if n_assigned > 0:
                logger.info(f"Core value counts:\n{adata.obs['Core'].value_counts(dropna=False)}")

            # Core ID 없는 세포 제거 (선택적 단계) - 필요시 주석 해제
            # n_before = adata.n_obs
            # adata = adata[adata.obs["Core"].notna()].copy()
            # n_after = adata.n_obs
            # if n_before != n_after:
            #    logger.info(f"Removed {n_before - n_after} cells with missing Core ID.")
            #    provenance_data['metadata']['log']['cells_removed_missing_core'] = n_before - n_after

            provenance_data['metadata']['params']['core_info_dir'] = core_info_dir
            provenance_data['metadata']['log']['n_core_files_processed'] = len(core_files)
            provenance_data['metadata']['log']['n_cells_assigned_core'] = n_assigned

            # 충돌 리포트 저장
            if conflict_report:
                conflict_df = pd.DataFrame(conflict_report)
                conflict_file = f"{output_prefix}_core_metadata_conflicts.csv"
                try:
                    conflict_df.to_csv(conflict_file, index=False)
                    logger.warning(f"Metadata conflicts detected. Report saved to: {conflict_file}")
                except Exception as e:
                    logger.error(f"Failed to save conflict report: {e}")
    else:
        logger.info("Core info directory not provided. Skipping core info integration.")


    # <<<--- [수정 시작]: 문제 컬럼 타입 변환 --->>>
    problematic_cols_to_convert = ['Transcripts_core', 'Area (µm^2)_core']
    logger.info(f"Attempting to convert potentially problematic obs columns to string: {problematic_cols_to_convert}")
    for col_name in problematic_cols_to_convert:
        if col_name in adata.obs.columns:
            try:
                # 데이터 타입 확인 로그 추가
                original_dtype = str(adata.obs[col_name].dtype)
                logger.info(f"Converting column '{col_name}' (dtype: {original_dtype}) to string type (NA -> 'NA_placeholder')...")

                # NA/NaN 값을 특정 문자열로 대체 후 문자열로 변환
                adata.obs[col_name] = adata.obs[col_name].astype(object).fillna('NA_placeholder').astype(str)
                logger.info(f"Successfully converted '{col_name}' to string.")

                # 변환 후 타입 확인 로그 (선택 사항)
                # logger.info(f"New dtype for '{col_name}': {adata.obs[col_name].dtype}")

            except Exception as e:
                logger.warning(f"Could not convert column '{col_name}' to string: {e}")
                # 에러 발생 시 provenance에도 기록
                provenance_data.setdefault('metadata', {}).setdefault('log', {}).setdefault('conversion_errors', []).append(col_name)
        else:
            logger.debug(f"Column '{col_name}' not found in adata.obs, skipping conversion.")
    # <<<--- [수정 끝] --->>>
    # <<<--- [수정 시작]: 모든 provenance_data['metadata']... 를 provenance_data['metadata']... 로 변경 --->>>
    # 예시:
    provenance_data.setdefault('metadata', {}).setdefault('params', {})['core_info_dir'] = core_info_dir # setdefault로 안전하게 접근
    provenance_data.setdefault('metadata', {}).setdefault('log', {})['n_core_files_processed'] = len(core_files) if core_info_dir and 'core_files' in locals() else 0
    # ... 다른 metadata provenance 정보도 유사하게 변경 ...
    provenance_data.setdefault('metadata', {})['timestamp'] = pd.Timestamp.now().isoformat()
    # <<<--- [수정 끝] --->>>


    # --- 3. QC 메트릭 계산 ---
    # <<<--- [수정 시작]: provenance_data['qc']... 를 provenance_data['qc']... 로 변경 --->>>
    logger.info("--- Step 3: Calculating QC Metrics ---")
    provenance_data['qc'] = {'params': {}, 'log': {}}
    # ... QC 단계의 모든 provenance 정보 저장 로직을 provenance_data['qc']... 로 변경 ...
    provenance_data['qc']['timestamp'] = pd.Timestamp.now().isoformat()
    # <<<--- [수정 끝] --->>>

    # 유전자 유형 식별 (Mito: 사람 'MT-', 마우스 'mt-'. 유연하게 처리)
    adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.upper().str.startswith(('RPS', 'RPL'))
    # Hemoglobin 정규식 수정: 시작(^), P 제외 ([^(P)])
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', case=False, regex=True)

    logger.info(f"Identified {adata.var['mt'].sum()} mitochondrial genes.")
    logger.info(f"Identified {adata.var['ribo'].sum()} ribosomal genes.")
    logger.info(f"Identified {adata.var['hb'].sum()} hemoglobin genes.")

    # Scanpy QC 계산
    # log1p=True는 시각화에는 유용하지만, 필터링 기준(예: pct_counts_mt)은 원본 비율로 하는 것이 일반적
    # 여기서는 계산 후 pct_counts* 컬럼만 사용하므로 log1p=False로 계산하고 필요시 로그변환
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo', 'hb'], inplace=True, percent_top=None, log1p=False)
    adata.obs['pct_counts_mt'] = adata.obs['total_counts_mt'] / adata.obs['total_counts'] * 100
    adata.obs['pct_counts_ribo'] = adata.obs['total_counts_ribo'] / adata.obs['total_counts'] * 100
    adata.obs['pct_counts_hb'] = adata.obs['total_counts_hb'] / adata.obs['total_counts'] * 100

    # 샘플 레벨 QC 요약 (필터링 전 상태 기록)
    qc_summary_before_filter = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'median_total_counts': float(np.median(adata.obs['total_counts'])),
        'mean_total_counts': float(np.mean(adata.obs['total_counts'])),
        'median_n_genes_by_counts': float(np.median(adata.obs['n_genes_by_counts'])),
        'mean_n_genes_by_counts': float(np.mean(adata.obs['n_genes_by_counts'])),
        'median_pct_counts_mt': float(np.median(adata.obs['pct_counts_mt'].fillna(0))), # NA는 0으로 간주
        'mean_pct_counts_mt': float(np.mean(adata.obs['pct_counts_mt'].fillna(0))),
    }
    adata.uns['qc_summary_before_filter'] = qc_summary_before_filter
    logger.info(f"QC metrics calculated. Summary (before filtering): {qc_summary_before_filter}")

    # QC 플롯 저장 (필터링 전 분포 확인용)
    if save_plots:
        try:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            sc.pl.violin(adata, keys='total_counts', jitter=0.4, ax=axs[0], show=False)
            sc.pl.violin(adata, keys='n_genes_by_counts', jitter=0.4, ax=axs[1], show=False)
            sc.pl.violin(adata, keys='pct_counts_mt', jitter=0.4, ax=axs[2], show=False)
            plt.suptitle(f"{sample_id} - QC Before Filtering", y=1.02)
            plt.tight_layout()
            plot_file = f"{output_prefix}_qc_violin_before_filter.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved QC violin plot (before filtering) to: {plot_file}")
        except Exception as e:
            logger.warning(f"Could not generate QC violin plot: {e}")

    provenance_data['qc']['timestamp'] = pd.Timestamp.now().isoformat()


    # --- 4. QC 필터링 (선택 사항) ---
    logger.info("--- Step 4: QC Filtering (Optional) ---")
    provenance_data['filter'] = {'params': {}, 'log': {}}
    provenance_data['filter']['params']['qc_filter_enabled'] = qc_filter

    if qc_filter:
        n_obs_before, n_vars_before = adata.shape
        logger.info(f"Applying QC filters with parameters: min_counts={min_counts_per_cell}, "
                    f"max_counts={max_counts_per_cell}, min_genes={min_genes_per_cell}, "
                    f"max_genes={max_genes_per_cell}, min_cells={min_cells_per_gene}, "
                    f"max_pct_mito={max_pct_mito}")

        # 필터링 조건 기록
        provenance_data['filter']['params']['min_counts_per_cell'] = min_counts_per_cell
        provenance_data['filter']['params']['max_counts_per_cell'] = max_counts_per_cell
        provenance_data['filter']['params']['min_genes_per_cell'] = min_genes_per_cell
        provenance_data['filter']['params']['max_genes_per_cell'] = max_genes_per_cell
        provenance_data['filter']['params']['min_cells_per_gene'] = min_cells_per_gene
        provenance_data['filter']['params']['max_pct_mito'] = max_pct_mito

        # 유전자 필터링 (먼저 수행하는 것이 일반적)
        if min_cells_per_gene is not None and min_cells_per_gene > 0:
            sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
            logger.info(f"Filtered genes: {n_vars_before - adata.n_vars} genes removed (< {min_cells_per_gene} cells).")

        # 세포 필터링
        filter_masks = []
        if min_genes_per_cell is not None:
            mask, _ = sc.pp.filter_cells(adata, min_genes=min_genes_per_cell, inplace=False)
            filter_masks.append(mask)
            logger.info(f"Cells to remove based on min_genes ({min_genes_per_cell}): {np.sum(~mask)}")
        if max_genes_per_cell is not None:
             mask = adata.obs['n_genes_by_counts'] <= max_genes_per_cell
             filter_masks.append(mask)
             logger.info(f"Cells to remove based on max_genes ({max_genes_per_cell}): {np.sum(~mask)}")
        if min_counts_per_cell is not None:
            mask, _ = sc.pp.filter_cells(adata, min_counts=min_counts_per_cell, inplace=False)
            filter_masks.append(mask)
            logger.info(f"Cells to remove based on min_counts ({min_counts_per_cell}): {np.sum(~mask)}")
        if max_counts_per_cell is not None:
            mask = adata.obs['total_counts'] <= max_counts_per_cell
            filter_masks.append(mask)
            logger.info(f"Cells to remove based on max_counts ({max_counts_per_cell}): {np.sum(~mask)}")
        if max_pct_mito is not None:
            # 결측치(mt 유전자 없거나 total_counts 0인 경우)는 통과시키도록 처리
            mask = (adata.obs['pct_counts_mt'] <= max_pct_mito) | adata.obs['pct_counts_mt'].isna()
            filter_masks.append(mask)
            logger.info(f"Cells to remove based on max_pct_mito ({max_pct_mito}%): {np.sum(~mask & adata.obs['pct_counts_mt'].notna())}") # NA 아닌 것만 카운트

        # 모든 마스크 조합
        if filter_masks:
            combined_mask = np.all(filter_masks, axis=0)
            n_removed = np.sum(~combined_mask)
            adata._inplace_subset_obs(combined_mask) # 내부 함수지만 효율적
            logger.info(f"Filtered cells: {n_removed} cells removed based on combined criteria.")
        else:
            logger.info("No cell filtering criteria applied.")

        logger.info(f"AnnData shape after QC filtering: {adata.shape}")
        provenance_data['filter']['log'] = {
            'n_obs_before': n_obs_before,
            'n_vars_before': n_vars_before,
            'n_obs_after': adata.n_obs,
            'n_vars_after': adata.n_vars,
            'n_cells_removed': n_obs_before - adata.n_obs,
            'n_genes_removed': n_vars_before - adata.n_vars
        }

        # QC 플롯 저장 (필터링 후)
        if save_plots and (n_obs_before != adata.n_obs or n_vars_before != adata.n_vars):
            try:
                fig, axs = plt.subplots(1, 3, figsize=(15, 4))
                sc.pl.violin(adata, keys='total_counts', jitter=0.4, ax=axs[0], show=False)
                sc.pl.violin(adata, keys='n_genes_by_counts', jitter=0.4, ax=axs[1], show=False)
                sc.pl.violin(adata, keys='pct_counts_mt', jitter=0.4, ax=axs[2], show=False)
                plt.suptitle(f"{sample_id} - QC After Filtering", y=1.02)
                plt.tight_layout()
                plot_file = f"{output_prefix}_qc_violin_after_filter.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved QC violin plot (after filtering) to: {plot_file}")
            except Exception as e:
                 logger.warning(f"Could not generate QC violin plot after filtering: {e}")

    else:
        logger.info("QC filtering step skipped.")
        provenance_data['filter']['log'] = {'message': 'Filtering skipped'}

    provenance_data['filter']['timestamp'] = pd.Timestamp.now().isoformat()

    # 필터링 후 데이터가 비어있는지 확인
    if adata.n_obs == 0 or adata.n_vars == 0:
        logger.error("AnnData object became empty after filtering. Aborting.")
        return None

    # --- .raw 설정: rank_genes_groups 등에서 사용하기 위해 정규화/로그변환된 데이터 저장 ---
    # SCTransform을 사용하더라도, rank_genes는 lognorm 기반 데이터에서 수행하는 경우가 많음
    # 따라서 두 경우 모두 lognorm 계산을 수행하여 .raw에 저장
    logger.info("Preparing .raw attribute with log-normalized data.")
    adata_raw = adata.copy()
    sc.pp.normalize_total(adata_raw, target_sum=lognorm_target_sum)
    sc.pp.log1p(adata_raw)
    adata.raw = adata_raw # .raw에는 필터링 후, lognorm된 전체 유전자 데이터가 저장됨
    logger.info("Saved log-normalized data to adata.raw")


    # --- 5. 정규화 & 고분산 유전자(HVG) 선택 ---
    logger.info(f"--- Step 5: Normalization & HVG Selection ({normalization_method}) ---")
    provenance_data['normalization'] = {'params': {'method': normalization_method}, 'log': {}}
    provenance_data['hvg'] = {'params': {}, 'log': {}}

    if normalization_method == 'lognorm':
        logger.info(f"Applying LogNormalize: target_sum={lognorm_target_sum}")
        # .raw 준비 단계에서 이미 normalize, log1p 수행했으므로, 현재 adata에 적용
        sc.pp.normalize_total(adata, target_sum=lognorm_target_sum)
        adata.layers['lognorm_counts'] = adata.X.copy() # 정규화 결과 저장
        logger.info("Applying log1p transformation.")
        sc.pp.log1p(adata)
        adata.layers['log1p_counts'] = adata.X.copy() # 로그 변환 결과 저장

        provenance_data['normalization']['params']['lognorm_target_sum'] = lognorm_target_sum

        logger.info("Finding Highly Variable Genes (HVGs) using Scanpy method.")
        hvg_params = {
            'min_mean': lognorm_hvg_min_mean,
            'max_mean': lognorm_hvg_max_mean,
            'min_disp': lognorm_hvg_min_disp,
            'n_top_genes': lognorm_hvg_n_top_genes,
            'flavor': 'seurat_v3' if lognorm_hvg_n_top_genes is None else 'cell_ranger' # n_top_genes 유무에 따라 flavor 변경
        }
        # None 파라미터 제거
        hvg_params = {k: v for k, v in hvg_params.items() if v is not None}

        sc.pp.highly_variable_genes(adata, **hvg_params)
        n_hvg = adata.var['highly_variable'].sum()
        logger.info(f"Found {n_hvg} HVGs using parameters: {hvg_params}")
        provenance_data['hvg']['params'] = hvg_params
        provenance_data['hvg']['log']['n_hvg'] = n_hvg

        if save_plots:
            try:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                sc.pl.highly_variable_genes(adata, ax=ax, show=False)
                plot_file = f"{output_prefix}_hvg_scatter.png"
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved HVG scatter plot to: {plot_file}")
            except Exception as e:
                 logger.warning(f"Could not generate HVG scatter plot: {e}")

    elif normalization_method == 'sct':
        try:
            import sctransform
        except ImportError:
            logger.error("sctransform package is not installed. Please install it to use normalization_method='sct'. `pip install sctransform`")
            return None

        logger.info("Applying SCTransform.")
        # SCTransform은 raw counts (또는 QC 필터링된 raw counts)를 입력으로 받음
        # counts 레이어를 사용 (Step 1에서 복사해둠)
        if 'counts' not in adata.layers:
             logger.error("Raw counts layer 'counts' not found. Cannot run SCTransform.")
             return None

        counts_matrix = adata.layers['counts'].copy()
        # SCTransform은 scipy sparse matrix를 선호
        if not sparse.issparse(counts_matrix):
             counts_matrix = sparse.csr_matrix(counts_matrix)

        sct_params = {
            'gene_names': adata.var_names.tolist(),
            'cell_names': adata.obs_names.tolist(),
            'method': 'poisson', # 기본값
            'n_genes': sct_n_genes,
            'min_cells': sct_min_cells,
            'residual_type': sct_residual_type,
            'random_seed': random_seed, # 재현성
            'verbosity': sc.settings.verbosity # Scanpy와 동일한 로그 레벨 사용
        }
        logger.info(f"Running sctransform.vst with parameters: {sct_params}")

        try:
            # SCTransform 수행
            vst_out = sctransform.vst(**sct_params)

            # 결과 저장
            logger.info("Storing SCTransform results in AnnData object.")
            # 1. 잔차 (Residuals): PCA 등 다운스트림 분석에 사용 (HVG만 사용)
            adata.layers['sct_residuals'] = vst_out['residuals']

            # 2. 보정된 Count (Corrected Counts): 시각화, DE 분석 등에 사용 가능
            # vst_out['corrected_counts']는 dense일 수 있으므로 메모리 확인 필요
            if 'corrected_counts' in vst_out:
                 adata.layers['sct_corrected_counts'] = vst_out['corrected_counts']
                 logger.info("Stored 'sct_corrected_counts' layer.")

            # 3. SCTransform 모델 파라미터 저장 (선택적)
            adata.uns['sct_model_params'] = vst_out.get('model_pars', None)
            adata.uns['sct_model_params_fit'] = vst_out.get('model_pars_fit', None)

            # 4. 유전자 속성 (HVG 정보 포함) 병합
            sct_gene_attr = vst_out['gene_attr'].set_index('gene_name')
            # 이름 충돌 방지
            sct_gene_attr = sct_gene_attr.rename(columns={
                'residual_variance': 'sct_residual_variance',
                'highly_variable': 'sct_highly_variable' # 원래 highly_variable 컬럼과 구분
            })
            adata.var = adata.var.join(sct_gene_attr, how='left')

            # HVG 플래그 설정
            if 'sct_highly_variable' in adata.var.columns:
                 # sctransform 최신 버전은 highly_variable 컬럼을 직접 제공 (True/False)
                 adata.var['highly_variable'] = adata.var['sct_highly_variable'].fillna(False)
                 n_hvg = adata.var['highly_variable'].sum()
                 logger.info(f"Identified {n_hvg} HVGs using SCTransform (based on 'sct_highly_variable' flag).")
            else:
                 # 이전 버전 또는 플래그가 없는 경우 residual variance 기준 상위 N개 사용
                 logger.warning("'sct_highly_variable' flag not found in SCTransform output. Using top N genes based on residual variance.")
                 hvg_genes = sct_gene_attr.nlargest(sct_n_genes, 'sct_residual_variance').index
                 adata.var['highly_variable'] = adata.var_names.isin(hvg_genes)
                 n_hvg = adata.var['highly_variable'].sum()
                 logger.info(f"Flagged top {n_hvg} genes as HVGs based on SCTransform residual variance.")

            provenance_data['normalization']['params']['sct_params'] = sct_params
            provenance_data['hvg']['params'] = {'method': 'sctransform', 'n_genes_target': sct_n_genes}
            provenance_data['hvg']['log'] = {'n_hvg': n_hvg}

            # SCTransform 사용 시, PCA 입력으로 사용할 데이터를 결정해야 함.
            # 보통 HVG에 대한 잔차(residuals)를 사용함.
            # adata.X를 잔차로 설정할 수 있으나, 혼동을 줄이기 위해 레이어를 명시적으로 사용하는 것이 좋음.
            # 다음 단계(PCA)에서 sct_residuals 레이어를 사용하도록 처리.

        except Exception as e:
             logger.error(f"SCTransform failed: {e}")
             logger.error("Falling back to lognorm method.")
             # LogNorm으로 대체 시도 (위의 lognorm 코드 블록 재실행 또는 별도 처리 필요)
             # 여기서는 간단히 에러 로그 남기고 종료
             return None

    else:
        logger.error(f"Invalid normalization_method: {normalization_method}. Choose 'lognorm' or 'sct'.")
        return None

    provenance_data['normalization']['timestamp'] = pd.Timestamp.now().isoformat()
    provenance_data['hvg']['timestamp'] = pd.Timestamp.now().isoformat()

    # HVG 확인
    if 'highly_variable' not in adata.var.columns or adata.var['highly_variable'].sum() == 0:
        logger.error("No Highly Variable Genes were identified. Cannot proceed with downstream analysis.")
        return None

    # --- 6. 스케일링 (Scaling) ---
    # LogNorm 방식에서만 필요. SCTransform은 잔차가 이미 스케일링된 효과를 가짐.
    logger.info("--- Step 6: Scaling Data ---")
    provenance_data['scaling'] = {'params': {}, 'log': {}}

    if normalization_method == 'lognorm':
        logger.info(f"Scaling data to zero mean and unit variance (max_value={scale_max_value}).")
        # HVG만 사용하여 스케일링 (일반적인 방법)
        adata_hvg = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata_hvg, max_value=scale_max_value, zero_center=True) # zero_center=True 기본값

        # 스케일링된 결과를 원래 AnnData 객체의 .X에 반영 (주의: HVG만 남게 됨)
        # 또는 별도 레이어에 저장하고 PCA 시 해당 레이어 사용
        # 여기서는 .X를 업데이트하고, 원본 log1p 데이터는 .raw 또는 레이어에 있음
        adata = adata[:, adata.var.highly_variable].copy() # HVG로 부분집합 생성
        adata.X = adata_hvg.X # 스케일링된 데이터로 X 교체
        adata.layers['scaled_lognorm_counts'] = adata.X.copy() # 스케일링 결과 저장

        logger.info("Applied scaling to HVGs. adata.X now contains scaled log-normalized counts for HVGs.")
        provenance_data['scaling']['params'] = {'method': 'scanpy.pp.scale', 'max_value': scale_max_value}
        provenance_data['scaling']['log'] = {'subset_to_hvg': True, 'n_hvg_scaled': adata.n_vars}
    elif normalization_method == 'sct':
        logger.info("Scaling is implicitly handled by SCTransform (residuals). Skipping explicit scaling step.")
        # PCA에서 sct_residuals 레이어를 사용하므로 별도 스케일링 불필요
        provenance_data['scaling']['params'] = {'method': 'sctransform_residuals'}
        provenance_data['scaling']['log'] = {'message': 'Skipped, using SCT residuals'}

    provenance_data['scaling']['timestamp'] = pd.Timestamp.now().isoformat()


    # --- 7. 차원 축소 (PCA) ---
    logger.info("--- Step 7: Dimensionality Reduction (PCA) ---")
    provenance_data['pca'] = {'params': {}, 'log': {}}

    pca_params = {'n_comps': pca_n_comps, 'svd_solver': 'arpack', 'random_state': random_seed}
    logger.info(f"Running PCA with parameters: {pca_params}")

    if normalization_method == 'lognorm':
        # .X에 스케일링된 HVG 데이터가 있음
        sc.tl.pca(adata, **pca_params)
        logger.info("PCA performed on scaled log-normalized HVG data.")
        provenance_data['pca']['params'] = {**pca_params, 'input_data': 'scaled_lognorm_hvg'}
    elif normalization_method == 'sct':
        # SCTransform 잔차(HVG만)를 사용하여 PCA 수행
        if 'sct_residuals' not in adata.layers:
            logger.error("SCTransform residuals layer ('sct_residuals') not found. Cannot run PCA.")
            return None

        logger.info("Performing PCA on SCTransform residuals for HVGs.")
        # HVG에 대한 잔차만 사용
        adata_hvg_idx = adata.var['highly_variable']
        pca_input = adata.layers['sct_residuals'][:, adata_hvg_idx]

        # sc.tl.pca는 AnnData 객체를 받으므로 임시 객체 사용 또는 직접 계산
        # 여기서는 직접 계산 방식 사용 (더 효율적일 수 있음)
        from sklearn.decomposition import PCA
        pca_model = PCA(n_components=pca_n_comps, svd_solver='arpack', random_state=random_seed)

        # SCT 잔차는 sparse일 수 있으므로 dense 변환 필요 시 수행
        if sparse.issparse(pca_input):
            pca_input_dense = pca_input.toarray()
        else:
            pca_input_dense = pca_input

        adata.obsm['X_pca'] = pca_model.fit_transform(pca_input_dense)
        adata.uns['pca'] = {} # Scanpy 호환성을 위해 pca uns 정보 초기화
        adata.uns['pca']['variance_ratio'] = pca_model.explained_variance_ratio_
        adata.uns['pca']['variance'] = pca_model.explained_variance_
        # adata.varm['PCs'] 저장 필요 시: pca_model.components_.T 를 사용해야 함.
        # sc.tl.pca는 varm['PCs']를 자동으로 저장해주지만, 여기서는 수동 작업 필요.
        # 일단 varm['PCs'] 저장은 생략 (필요 시 추가)

        logger.info("PCA performed on SCTransform residuals for HVGs.")
        provenance_data['pca']['params'] = {**pca_params, 'input_data': 'sct_residuals_hvg'}

    provenance_data['pca']['log'] = {'n_comps_computed': pca_n_comps}

    # Elbow plot 저장
    if save_plots:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            sc.pl.pca_variance_ratio(adata, n_pcs=min(pca_n_comps, 50), log=True, ax=ax, show=False)
            plot_file = f"{output_prefix}_pca_variance_ratio.png"
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved PCA variance ratio plot to: {plot_file}")
        except Exception as e:
             logger.warning(f"Could not generate PCA variance ratio plot: {e}")

    provenance_data['pca']['timestamp'] = pd.Timestamp.now().isoformat()

    # 사용할 PC 개수 확인
    if neighbors_n_pcs > adata.obsm['X_pca'].shape[1]:
        logger.warning(f"neighbors_n_pcs ({neighbors_n_pcs}) is greater than the number of computed PCs ({adata.obsm['X_pca'].shape[1]}). Adjusting to use all computed PCs.")
        neighbors_n_pcs = adata.obsm['X_pca'].shape[1]


    # --- 8. 이웃 그래프 생성 (Neighborhood Graph) ---
    logger.info("--- Step 8: Building Neighborhood Graph ---")
    provenance_data['neighbors'] = {'params': {}, 'log': {}}

    neighbors_params = {
        'n_neighbors': neighbors_n_neighbors,
        'n_pcs': neighbors_n_pcs,
        'use_rep': 'X_pca', # PCA 결과를 사용 명시
        'random_state': random_seed
    }
    logger.info(f"Building KNN graph using parameters: {neighbors_params}")
    sc.pp.neighbors(adata, **neighbors_params)

    provenance_data['neighbors']['params'] = neighbors_params
    provenance_data['neighbors']['log'] = {'graph_built': True}
    provenance_data['neighbors']['timestamp'] = pd.Timestamp.now().isoformat()


    # --- 9. 클러스터링 ---
    logger.info("--- Step 9: Clustering ---")
    provenance_data['clustering'] = {'params': {}, 'log': {}}

    cluster_key = f"clusters_{cluster_algo}_res{cluster_resolution}"
    cluster_params = {'resolution': cluster_resolution, 'key_added': cluster_key, 'random_state': random_seed}

    logger.info(f"Running {cluster_algo} clustering with parameters: {cluster_params}")

    if cluster_algo == 'leiden':
        sc.tl.leiden(adata, **cluster_params)
    elif cluster_algo == 'louvain':
        sc.tl.louvain(adata, **cluster_params)
    else:
        logger.error(f"Unsupported clustering algorithm: {cluster_algo}. Choose 'leiden' or 'louvain'.")
        return None

    n_clusters = len(adata.obs[cluster_key].cat.categories)
    logger.info(f"Found {n_clusters} clusters using {cluster_algo} (resolution={cluster_resolution}). Results stored in adata.obs['{cluster_key}'].")

    provenance_data['clustering']['params'] = {**cluster_params, 'algorithm': cluster_algo}
    provenance_data['clustering']['log'] = {'cluster_key': cluster_key, 'n_clusters': n_clusters}
    provenance_data['clustering']['timestamp'] = pd.Timestamp.now().isoformat()


    # --- 10. 시각화 (UMAP) ---
    logger.info("--- Step 10: Visualization (UMAP) ---")
    provenance_data['umap'] = {'params': {}, 'log': {}}

    umap_params = {'min_dist': umap_min_dist, 'spread': umap_spread, 'random_state': random_seed}
    logger.info(f"Running UMAP with parameters: {umap_params}")
    sc.tl.umap(adata, **umap_params)

    provenance_data['umap']['params'] = umap_params
    provenance_data['umap']['log'] = {'umap_computed': True}

    # UMAP 플롯 저장 (클러스터별 색상)
    if save_plots:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            # legend 를 plot 바깥으로 빼거나, 개수가 많으면 안 보이게 처리
            legend_loc = 'on data' if n_clusters <= 20 else None
            sc.pl.umap(adata, color=[cluster_key], ax=ax, show=False,
                      legend_loc=legend_loc, legend_fontsize=8 if legend_loc else None,
                      title=f"{cluster_key} ({n_clusters} clusters)")
            plot_file = f"{output_prefix}_umap_{cluster_key}.png"
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved UMAP plot colored by '{cluster_key}' to: {plot_file}")

            # 공간 플롯 저장 (클러스터별 색상) - spatial 정보가 있는 경우
            if 'spatial' in adata.obsm:
                 fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                 sc.pl.spatial(adata, color=[cluster_key], ax=ax, show=False, spot_size=20, # spot_size 조절 필요
                               legend_loc=legend_loc, legend_fontsize=8 if legend_loc else None,
                               title=f"{cluster_key} ({n_clusters} clusters) - Spatial")
                 plot_file = f"{output_prefix}_spatial_{cluster_key}.png"
                 fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                 plt.close(fig)
                 logger.info(f"Saved spatial plot colored by '{cluster_key}' to: {plot_file}")

        except Exception as e:
             logger.warning(f"Could not generate UMAP/Spatial plots: {e}")

    provenance_data['umap']['timestamp'] = pd.Timestamp.now().isoformat()


    # --- 11. 마커 유전자 탐색 (Rank Genes Groups) ---
    logger.info("--- Step 11: Finding Marker Genes ---")
    provenance_data['rank_genes'] = {'params': {}, 'log': {}}

    rank_genes_key = f"{cluster_key}_rank_genes_{rank_genes_method}"
    rank_genes_params = {
        'groupby': cluster_key,
        'method': rank_genes_method,
        'key_added': rank_genes_key,
        'use_raw': True, # .raw 사용 (log-normalized, non-scaled, all genes)
        'pts': rank_genes_pts,
        'n_genes': -1 # 모든 유전자에 대해 계산 (나중에 상위 N개 선택)
    }
    logger.info(f"Running rank_genes_groups with parameters: {rank_genes_params}")

    try:
        sc.tl.rank_genes_groups(adata, **rank_genes_params)
        logger.info(f"Marker gene analysis completed. Results stored in adata.uns['{rank_genes_key}'].")
        provenance_data['rank_genes']['params'] = rank_genes_params
        provenance_data['rank_genes']['log'] = {'rank_genes_key': rank_genes_key}

        # 마커 유전자 플롯 저장
        if save_plots:
            try:
                fig = sc.pl.rank_genes_groups(adata, n_genes=rank_genes_n_genes, key=rank_genes_key, sharey=False, show=False, figsize=(max(5, n_clusters * 1.5), 10)) # 너비 동적 조절
                plot_file = f"{output_prefix}_rank_genes_{cluster_key}.png"
                # plot 객체가 figure 리스트를 반환할 수 있으므로 확인
                if isinstance(fig, list): # matplotlib axes list
                    plt.gcf().savefig(plot_file, dpi=150, bbox_inches='tight')
                    plt.close(plt.gcf())
                elif fig: # matplotlib figure object
                    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else: # show=False 시 None 반환 가능
                    # 다시 그려서 저장
                     sc.pl.rank_genes_groups(adata, n_genes=rank_genes_n_genes, key=rank_genes_key, sharey=False, show=False, save=f"_rank_genes_{cluster_key}.png")
                     # scanpy가 저장한 파일명 변경
                     default_filename = f"./figures/rank_genes_groups_{cluster_key}.png" # scanpy 기본 저장 경로/파일명
                     if os.path.exists(default_filename):
                         os.rename(default_filename, plot_file)
                         # figures 디렉토리 비었으면 삭제
                         if not os.listdir("./figures"):
                              os.rmdir("./figures")
                     else:
                          logger.warning("Failed to locate saved rank_genes_groups plot.")


                logger.info(f"Saved rank_genes_groups plot to: {plot_file}")
            except Exception as e:
                logger.warning(f"Could not generate rank_genes_groups plot: {e}")

    except Exception as e:
        logger.error(f"rank_genes_groups failed: {e}")
        provenance_data['rank_genes']['log'] = {'error': str(e)}


    provenance_data['rank_genes']['timestamp'] = pd.Timestamp.now().isoformat()

    # --- 12. 최종 결과 저장 ---
    logger.info("--- Step 12: Saving Results ---")
    # <<<--- [수정 시작]: provenance 저장 관련 코드 추가/변경 --->>>

    # QC 요약 (필터링 후)
    qc_summary_after_filter = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'median_total_counts': float(np.median(adata.obs['total_counts'])),
        'mean_total_counts': float(np.mean(adata.obs['total_counts'])),
        'median_n_genes_by_counts': float(np.median(adata.obs['n_genes_by_counts'])),
        'mean_n_genes_by_counts': float(np.mean(adata.obs['n_genes_by_counts'])),
        'median_pct_counts_mt': float(np.median(adata.obs['pct_counts_mt'].fillna(0))),
        'mean_pct_counts_mt': float(np.mean(adata.obs['pct_counts_mt'].fillna(0))),
    }
    provenance_data['final_qc_summary'] = qc_summary_after_filter
    logger.info(f"Final QC Summary (also logged in provenance): {qc_summary_after_filter}")

    # --- Provenance 데이터 JSON 파일로 저장 ---
    provenance_file = f"{output_prefix}_provenance.json"
    logger.info(f"Saving provenance information to: {provenance_file}")
    try:
        import json
        # JSON으로 직렬화하기 전에 NumPy 타입을 Python 기본 타입으로 변환하는 함수 (필요 시)
        def convert_numpy_types(obj):
             if isinstance(obj, np.integer):
                 return int(obj)
             elif isinstance(obj, np.floating):
                 return float(obj)
             elif isinstance(obj, np.ndarray):
                 return obj.tolist() # 리스트로 변환
             elif isinstance(obj, (pd.Timestamp, pd.Timedelta)): # Timestamp 등도 문자열로
                 return str(obj)
             # 다른 필요한 타입 변환 추가 가능
             # raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable") # 변환 불가 시 에러

        with open(provenance_file, 'w') as f:
             # default=convert_numpy_types 인자를 사용하여 NumPy 타입을 변환
             json.dump(provenance_data, f, indent=4, default=convert_numpy_types)
        logger.info("Provenance data saved successfully.")
    except TypeError as e:
        logger.error(f"Failed to serialize provenance data to JSON: {e}. Saving skipped. Check for non-standard data types.")
    except Exception as e:
        logger.error(f"Failed to save provenance data to JSON: {e}. Saving skipped.")
    # --- Provenance 저장 끝 ---

    if save_adata:
        adata_file = f"{output_prefix}_processed.h5ad"
        logger.info(f"Attempting to save final AnnData object to: {adata_file}")
        try:
            # .uns 에서 provenance는 이미 제거되었으므로 신경 쓸 필요 없음
            # 다른 잠재적 문제 객체 제거 (SCT 모델 파라미터 등)
            if 'sct_model_params' in adata.uns: del adata.uns['sct_model_params']
            if 'sct_model_params_fit' in adata.uns: del adata.uns['sct_model_params_fit']

            # AnnData 객체 저장
            adata.write(adata_file, compression='gzip')
            logger.info(f"Successfully saved processed AnnData object to: {adata_file}")

        except Exception as e:
            logger.error(f"Failed to save AnnData object: {e}")
            # 이제 provenance_data 에 직접 에러 로그를 남길 수 없음 (저장 시점 이후)
            # 에러는 기본 로그 파일에 기록됨
    else:
        logger.info("Skipping final AnnData object saving as requested.")

    # <<<--- [수정 끝] --->>>

    logger.info("Preprocessing pipeline finished.")

    # 로깅 핸들러 정리
    logger.removeHandler(file_handler)
    file_handler.close()
    # 콘솔 핸들러는 유지 (다른 로거에서 사용할 수 있음)

    return adata

# --- 함수 사용 예시 ---
if __name__ == '__main__':
    # 실제 파일 경로와 원하는 파라미터로 수정하세요.
    # 가상 경로 예시
    h5_path = "/path/to/your/data/cell_feature_matrix.h5"
    cells_path = "/path/to/your/data/cells.parquet"
    boundaries_path = "/path/to/your/data/cell_boundaries.parquet"
    core_dir = "/path/to/your/core_csv_files/" # Core 정보 파일이 없다면 None
    output_prefix = "/path/to/output/your_sample_id" # 출력 경로 및 파일명 접두사
    sample_name = "CRC_Patient1_Region1"

    # --- 예시: 기본 LogNorm 사용 ---
    # try:
    #     adata_processed_lognorm = preprocess_xenium(
    #         h5_url=h5_path,
    #         cells_parquet_url=cells_path,
    #         boundaries_parquet_url=boundaries_path, # 현재 미사용
    #         core_info_dir=core_dir,
    #         output_prefix=f"{output_prefix}_lognorm",
    #         sample_id=sample_name,
    #         normalization_method='lognorm',
    #         # --- 필요에 따라 다른 파라미터 수정 ---
    #         qc_filter=True,
    #         max_pct_mito=15.0,
    #         neighbors_n_pcs=30,
    #         cluster_resolution=0.6,
    #         save_plots=True,
    #         save_adata=True
    #     )
    #     if adata_processed_lognorm:
    #         print("\nLogNorm preprocessing completed.")
    #         print("Final AnnData info:")
    #         print(adata_processed_lognorm)
    #         # 추가 분석 수행 가능
    #     else:
    #         print("\nLogNorm preprocessing failed.")

    # except Exception as e:
    #      print(f"\nAn error occurred during the example run (LogNorm): {e}")


    # --- 예시: SCTransform 사용 ---
    # try:
    #     adata_processed_sct = preprocess_xenium(
    #         h5_url=h5_path,
    #         cells_parquet_url=cells_path,
    #         boundaries_parquet_url=boundaries_path, # 현재 미사용
    #         core_info_dir=core_dir,
    #         output_prefix=f"{output_prefix}_sct",
    #         sample_id=sample_name,
    #         normalization_method='sct',
    #         # --- SCTransform 관련 파라미터 조정 가능 ---
    #         sct_n_genes=3000,
    #         # --- 필요에 따라 다른 파라미터 수정 ---
    #         qc_filter=True,
    #         max_pct_mito=15.0,
    #         neighbors_n_pcs=30, # SCT는 보통 더 적은 PC 사용 가능
    #         cluster_resolution=0.5,
    #         save_plots=True,
    #         save_adata=True
    #     )
    #     if adata_processed_sct:
    #         print("\nSCTransform preprocessing completed.")
    #         print("Final AnnData info:")
    #         print(adata_processed_sct)
    #         # 추가 분석 수행 가능
    #     else:
    #          print("\nSCTransform preprocessing failed.")

    # except Exception as e:
    #      print(f"\nAn error occurred during the example run (SCT): {e}")

    print("\nExample usage script finished. Uncomment and modify the try-except blocks with your actual file paths and parameters to run the preprocessing.")




