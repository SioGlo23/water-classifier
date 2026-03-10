import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path

import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import label as ndi_label
from scipy.ndimage import sum_labels
from shapely.geometry import shape
from shapely.ops import unary_union

st.set_page_config(page_title="Классификация воды в карьере", layout="centered", page_icon="🛰️")
st.title("🛰️ Классификация воды в карьере")
st.markdown("Загрузите файлы → укажите параметры → получите готовый вектор")

# ====================== ФОРМА ======================
st.subheader("📤 Загрузка файлов")

raster_file = st.file_uploader("Снимок (RGBN) — .tif/.tiff", type=["tif", "tiff"])

quarry_files = st.file_uploader(
    "Вектор карьера (.shp + все файлы: shx, dbf, prj...)",
    type=["shp", "shx", "dbf", "prj", "cpg"], accept_multiple_files=True
)

training_files = st.file_uploader(
    "Эталоны (.shp + все файлы: shx, dbf, prj...)",
    type=["shp", "shx", "dbf", "prj", "cpg"], accept_multiple_files=True
)

st.subheader("⚙️ Параметры")
col1, col2 = st.columns(2)
with col1:
    water_class = st.number_input("Класс воды", value=9)
    smoothing_distance = st.number_input("Сглаживание границ (метры)", value=3.0, step=0.5)
with col2:
    quarry_class = st.number_input("Класс карьера", value=45)
    min_water_size = st.number_input("Удалять объекты воды меньше (пикселей)", value=30)

label_field = st.text_input("Имя поля с классом в эталонах", value="class", help="Например: class, Class, ID, grid и т.д.")

# ====================== ЗАПУСК ======================
if st.button("🚀 Запустить классификацию", type="primary", use_container_width=True):
    if not (raster_file and quarry_files and training_files):
        st.error("❌ Загрузите все файлы!")
        st.stop()

    with st.spinner("Обработка..."):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)

                # Сохраняем файлы
                raster_path = tmp / raster_file.name
                raster_path.write_bytes(raster_file.getbuffer())

                def save_files(file_list):
                    for f in file_list:
                        (tmp / f.name).write_bytes(f.getbuffer())

                save_files(quarry_files)
                save_files(training_files)

                # Находим .shp файлы
                shp_files = list(tmp.glob("*.shp"))
                if len(shp_files) < 2:
                    st.error("❌ Не найдено два .shp файла")
                    st.stop()

                # Определяем, какой шейпфайл — карьер, какой — эталоны (по количеству файлов в загрузчике)
                quarry_shp = next((f for f in shp_files if any(q.name.startswith(f.stem) for q in quarry_files)), shp_files[0])
                training_shp = next(f for f in shp_files if f != quarry_shp)

                # ====================== ОБРАБОТКА ======================
                with rasterio.open(raster_path) as src:
                    data = src.read()
                    transform = src.transform
                    height, width = src.height, src.width
                    meta = src.meta.copy()
                    nodata_val = src.nodata
                    crs = src.crs

                gdf_quarry = gpd.read_file(quarry_shp)
                gdf_train = gpd.read_file(training_shp)

                # === ИСПРАВЛЕНИЕ: проверка поля класса ===
                if label_field not in gdf_train.columns:
                    st.error(f"❌ Поле **'{label_field}'** не найдено в файле эталонов!\n\n"
                             f"**Доступные поля:** {list(gdf_train.columns)}")
                    st.stop()

                # Приводим проекции
                if gdf_quarry.crs != crs:
                    gdf_quarry = gdf_quarry.to_crs(crs)
                if gdf_train.crs != crs:
                    gdf_train = gdf_train.to_crs(crs)

                # Убираем пустые значения в поле класса
                if gdf_train[label_field].isna().any():
                    gdf_train = gdf_train[gdf_train[label_field].notna()].copy()

                # Маски и классификация (остальной код без изменений)
                inside_mask = ~geometry_mask(list(gdf_quarry.geometry), out_shape=(height, width),
                                             transform=transform, all_touched=True, invert=False)

                nodata_mask = (np.any(data == nodata_val, axis=0) if nodata_val is not None and not np.isnan(nodata_val)
                               else np.any(np.isnan(data), axis=0) if nodata_val is not None else np.zeros((height, width), dtype=bool))

                train_shapes = [(geom, int(label)) for geom, label in zip(gdf_train.geometry, gdf_train[label_field])]
                label_raster = rasterize(train_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

                valid_train_mask = (label_raster > 0) & inside_mask & ~nodata_mask
                clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
                clf.fit(data[:, valid_train_mask].T, label_raster[valid_train_mask])

                class_raster = np.zeros((height, width), dtype=np.uint8)
                class_raster[inside_mask & ~nodata_mask] = clf.predict(data[:, inside_mask & ~nodata_mask].T)

                class_raster[class_raster == 3] = 1

                # Чистка и векторизация (тот же код, что раньше)
                water_mask = (class_raster == 1) & (inside_mask & ~nodata_mask)
                labeled_array, num_features = ndi_label(water_mask)
                if num_features > 0:
                    sizes = sum_labels(water_mask.astype(int), labeled_array, index=np.arange(1, num_features + 1))
                    small = np.where(sizes < min_water_size)[0] + 1
                    if len(small) > 0:
                        class_raster[np.isin(labeled_array, small)] = 2

                shape_gen = shapes((class_raster == 1).astype(np.uint8), mask=(class_raster == 1), transform=transform)
                geoms = [shape(geom) for geom, val in shape_gen if val == 1]
                water_gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs) if geoms else gpd.GeoDataFrame(geometry=[], crs=crs)

                if len(water_gdf) > 0:
                    smoothed = water_gdf.geometry.buffer(smoothing_distance).buffer(-smoothing_distance).simplify(0.5)
                    water_gdf.geometry = smoothed
                    water_gdf = water_gdf[~water_gdf.geometry.is_empty]

                # Финальный вектор
                water_union = water_gdf.geometry.unary_union if len(water_gdf) > 0 else None
                final_geoms = []
                final_classes = []
                for geom in gdf_quarry.geometry:
                    diff = geom.difference(water_union) if water_union else geom
                    if not diff.is_empty:
                        final_geoms.extend([diff] if diff.geom_type == 'Polygon' else diff.geoms)
                        final_classes.append(quarry_class)
                for g in water_gdf.geometry:
                    final_geoms.append(g)
                    final_classes.append(water_class)

                final_gdf = gpd.GeoDataFrame(geometry=final_geoms, crs=crs)
                final_gdf['class'] = final_classes

                # Сохранение
                base_name = raster_file.name.rsplit('.', 1)[0]
                result_shp = tmp / f"{base_name}_class.shp"
                final_gdf.to_file(result_shp)

                zip_path = tmp / f"{base_name}_class.zip"
                with zipfile.ZipFile(zip_path, 'w') as z:
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        f = result_shp.with_suffix(ext)
                        if f.exists():
                            z.write(f, f.name)

                with open(zip_path, "rb") as f:
                    st.download_button(
                        "📥 Скачать результат (ZIP)",
                        data=f.read(),
                        file_name=f"{base_name}_class.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                st.success("✅ Готово!")
                st.balloons()

        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            st.info("Если ошибка повторяется — пришли скриншот.")

st.caption("Полностью готовый веб-инструмент. Работает в любом браузере.")