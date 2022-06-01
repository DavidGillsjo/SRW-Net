import json
import yaml
import scipy.io as sio
import os
import os.path as osp
import cv2
import skimage.draw
import scipy.ndimage
from itertools import combinations, combinations_with_replacement
import matplotlib
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import csv
import sys
import shutil
import shapely.geometry as sg
import shapely
import open3d
from scipy.spatial.distance import cdist
from copy import deepcopy
import cProfile, pstats
import multiprocessing as mp
import time
from datetime import timedelta
from tqdm import tqdm
import logging
import re
import errno
import random
from parsing.utils.visualization import ImagePlotter
from collections import deque
from descartes import PolygonPatch
import seaborn as sns
from structured3D_geometry import Plane, InvalidGeometryError, CMP_EPS, CMP_EPS2, MIN_LEN, DIV_EPS, normalize, parse_camera_info

ALL_SINGLE_LINE_CLASSES = (
    [frozenset(['invalid'])]  #Never output, just for training purposes.
    + [frozenset([a,b]) for a,b in combinations_with_replacement(['wall', 'floor', 'ceiling', 'window', 'door'],2)]
    )
ALL_SINGLE_LINE_CLASS_TO_IDX = {c:i for i,c in enumerate(ALL_SINGLE_LINE_CLASSES)}

REDUCED_SINGLE_LINE_CLASSES = (
    'invalid', #Never output, just for training purposes.
    'wall',
    'floor',
    'ceiling',
    'window',
    'door'
)
REDUCED_SINGLE_LINE_CLASS_TO_IDX = {c:i for i,c in enumerate(REDUCED_SINGLE_LINE_CLASSES)}

REDUCED_SINGLE_LINE_CLASSES_MAPPING = [0]*len(ALL_SINGLE_LINE_CLASSES)
for idx, label in enumerate(ALL_SINGLE_LINE_CLASSES):
    for r_idx in reversed(range(len(REDUCED_SINGLE_LINE_CLASSES))):
        if REDUCED_SINGLE_LINE_CLASSES[r_idx] in label:
            REDUCED_SINGLE_LINE_CLASSES_MAPPING[idx] = r_idx
            break

JUNCTION_CLASSES = (
    'invalid',
    'false',
    'proper'
)


P_IDX = 0
NBR_NEG_LINES = 40
PLANE_SEMANTIC_CLASSES = [
    'wall',
    'floor',
    'ceiling',
    'door',
    'window',
    'outwall'
    ]

IGNORE_CLASSES = [
    'outwall',
    # 'window',
    # 'door'
]

DEF_IDX_MATCH = {0:0, 1:1, None:None}

DATA_RANGE = {
    'train':(0,3000),
    'val':(3000, 3250),
    'test':(3250, 3500)
}

def plane2label(plane_label, sematic_label):
    if sematic_label in set(['door', 'window', 'outwall']):
        return sematic_label
    else:
        return plane_label

def link_and_annotate(root, scene_dir, out_image_dir, make_plot = False, invalid_rooms = None):
    logger = logging.getLogger('structured3D2wireframe')
    scene_id = scene_dir.split('_')[1]
    if invalid_rooms.scene_is_invalid(scene_id):
        raise InvalidGeometryError('Scene is on invalid list')

    with open(osp.join(root, scene_dir, 'annotation_3d.json')) as f:
        ann_3D = json.load(f)

    # Prepare ann_3D with information we want later
    ann_3D['lineJunctionMatrix'] = np.array(ann_3D['lineJunctionMatrix'], dtype=np.bool)
    ann_3D['planeLineMatrix'] = np.array(ann_3D['planeLineMatrix'], dtype=np.bool)
    ann_3D['junctionCoords'] = np.array([j['coordinate'] for j in ann_3D['junctions']], dtype=np.float32).T

    #Map semantics to planes and vice versa
    semantic2planeID = {s:[] for s in PLANE_SEMANTIC_CLASSES}
    for semantic in ann_3D['semantics']:
        for id in semantic['planeID']:
            assert 'semantic' not in ann_3D['planes'][id]
            p = ann_3D['planes'][id]
            label = plane2label(p['type'], semantic['type'])
            ann_3D['planes'][id]['semantic'] = label
            semantic2planeID[label].append(id)
    for type, planeIDs in semantic2planeID.items():
        semantic2planeID[type] = np.unique(planeIDs)

    render_dir = osp.join(root, scene_dir, '2D_rendering')
    ann_3D['planeList'] = Plane.create_list_from_ann3D(ann_3D, render_dir)#, scene_id == '00972')



    """ Assumption check, raise error if false """
    assert np.all([idx == plane['ID'] for (idx,plane) in enumerate(ann_3D['planes'])])
    assert np.all([idx == line['ID'] for (idx,line) in enumerate(ann_3D['lines'])])
    assert np.all([idx == junction['ID'] for (idx,junction) in enumerate(ann_3D['junctions'])])
    if not np.all([2 == mask.sum() for mask in ann_3D['lineJunctionMatrix']]):
        raise InvalidGeometryError('Some lines have more than 3 junctions.')


    filtered_line_idx = []
    ignore_ids = np.concatenate([semantic2planeID[c] for c in IGNORE_CLASSES])
    for l_id, line2junc in enumerate(ann_3D['lineJunctionMatrix']):
        plane_ids = np.flatnonzero(ann_3D['planeLineMatrix'][:,l_id])
        if np.any(np.isin(plane_ids, ignore_ids, assume_unique=True)):
            continue
        filtered_line_idx.append(l_id)

    if make_plot:
        plot_scene_dir = '/host_home/plots/hawp/{}'.format(scene_id)
        os.makedirs(plot_scene_dir, exist_ok=True)


    ann_3D['semantic2planeID'] = semantic2planeID
    ann_3D['filtered_line_idx'] = filtered_line_idx
    out_ann = []


    # print('Scene', scene_id)
    for room_id in os.listdir(render_dir):
        # print('Room', room_id)
        if invalid_rooms and invalid_rooms.room_is_invalid(scene_id, room_id):
            logger.info('Skipping Room {} in Scene {}, it is invalid'.format(room_id, scene_id))
            continue
        room_dir = osp.join(render_dir, room_id, 'perspective', 'full')
        for pos_id in os.listdir(room_dir):
            ann = {}
            pos_dir = osp.join(room_dir, pos_id)
            img_name = 'S{}R{:0>5s}P{}.png'.format(scene_id, room_id, pos_id)
            img = cv2.imread(osp.join(pos_dir,'rgb_rawlight.png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ann['filename'] = img_name
            ann['height'], ann['width'] = img.shape[:2]
            try:
                #Remove if existing
                os.remove(osp.abspath(osp.join(out_image_dir, img_name)))
            except IOError:
                pass
            os.symlink(
                osp.relpath(osp.join(pos_dir,'rgb_rawlight.png'), start = out_image_dir),
                osp.join(out_image_dir, img_name))

            with open(osp.join(pos_dir, 'camera_pose.txt')) as f:
                pose = next(csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC))

            edge2l_idx, line_ann = add_line_annotation(img, ann_3D, pose, scene_id)
            if line_ann is None:
                logger.warning('No junctions found for image {}, skipping'.format(img_name))
                continue

            ann.update(line_ann)

            if make_plot:
                plotter = ImagePlotter(REDUCED_SINGLE_LINE_CLASSES, JUNCTION_CLASSES)
                plotter.plot_gt_image(img, ann, plot_scene_dir, desc = 'pos')#, edges_text = edge2l_idx)
                plotter.plot_gt_image(img, ann, plot_scene_dir, desc = 'neg', use_negative_edges = True)

            out_ann.append(ann)

    return out_ann

def add_line_annotation(image, ann_3D, pose,scene_id=None):
    out_junctions = []
    out_junctions_semantic = []
    out_edges = []
    out_edges_semantic = []
    junc2edge = []
    edge2l_idx = []
    height, width = image.shape[:2]

    R, t, K = parse_camera_info(pose, width, height)
    K_inv = np.linalg.inv(K)
    T = np.block([
        [R, t],
        [np.array([0,0,0,1])]
    ])
    T_inv = np.linalg.inv(T)

    img_planes = [p.transform(T_inv) for p in ann_3D['planeList']]
    img_junctions = R@ann_3D['junctionCoords'] + t
    img_poly = sg.box(0,0,width, height)

    junctions = []
    edges_pos = []
    jidx_scene2img = {}
    # Intersect each line with image
    for l_idx in ann_3D['filtered_line_idx']:

        # Find junctions for line
        j_idx = np.flatnonzero(ann_3D['lineJunctionMatrix'][l_idx])
        j12_img = img_junctions[:,j_idx]

        modified, j12_img = line_to_front(j12_img)
        planes_mask = ann_3D['planeLineMatrix'][:,l_idx]

        # Check if line was in front of camera
        if j12_img is None:
            continue



        _, j12_img_px = line_to_img(j12_img, K, img_poly = img_poly)

        # Check if line was inside image bounds
        if j12_img_px is None:
            continue

        # print(j12_img)
        # j12_img = K_inv@np.vstack([j12_img_px, np.ones([1,2])])
        # print(j12_img)
        #Check occluding planes for overlap in image
        segment_list, modified_list, match_idx_list  = get_visible_segments(modified, j12_img, img_planes, planes_mask, img_junctions, l_idx = l_idx, ann_3D = ann_3D)
        if not segment_list:
            if l_idx == DBG_LIDX: print('No visible segments')
            continue

        # print('------------------ l_idx {} -----------------|'.format(l_idx))
        # print('Line semantics', [ann_3D['planes'][id]['semantic'] for id in np.flatnonzero(planes_mask)])
        # print('Modified line to front', modified)

        # j12_img_px = K@j12_img_visible
        # j12_img_px = j12_img_px[:2] / j12_img_px[2]
        modified_list_pix = []
        segment_list_pix = []
        match_idx_list_pix = []
        segment_list_z = []
        for mod, seg,match_idx in zip(modified_list, segment_list, match_idx_list):

            mod_pix, seg_pix = line_to_img(seg, K, img_poly = img_poly, l_idx=l_idx)
            # print('Modified to img', mod_pix)
            if seg_pix is not None:
                segment_list_pix.append(seg_pix)
                modified_list_pix.append(mod | mod_pix)
                match_idx_list_pix.append(match_idx)
                segment_list_z.append(seg[2])


        #Check if line was inside image bounds
        if not segment_list_pix:
            continue

        """ Store result
        Keep track of the junctions by using jidx_scene2img and the matches between
        original line junctions and visible segments.
        """
        all_semantics = frozenset([ann_3D['planes'][id]['semantic'] for id in np.flatnonzero(planes_mask)])
        edge_sem = ALL_SINGLE_LINE_CLASS_TO_IDX[all_semantics]

        if l_idx == DBG_LIDX: print('Nbr segments:', len(segment_list_pix))
        if l_idx == DBG_LIDX: print('Line', segment_list_pix)
        if l_idx == DBG_LIDX: print('Modified', modified_list_pix)
        if l_idx == DBG_LIDX: print('Match', match_idx_list)

        for mod, seg, match_idx in zip(modified_list_pix, segment_list_pix, match_idx_list_pix):
            edge = []
            for i in range(2):
                old_idx = match_idx[i]
                if mod[i]:
                    # Junction is fake, simply add the new position to list
                    edge.append(len(out_junctions))
                    junc2edge.append([len(out_edges)])
                    out_junctions.append(seg[:,i])
                    out_junctions_semantic.append(1)
                elif j_idx[old_idx] in jidx_scene2img:
                    # We have already added this junction
                    out_j_idx = jidx_scene2img[j_idx[old_idx]]
                    edge.append(out_j_idx)
                    junc2edge[out_j_idx].append(len(out_edges))
                else:
                    # New junction, create mapping and add to list
                    jidx_scene2img[j_idx[old_idx]] = len(out_junctions)
                    edge.append(len(out_junctions))
                    junc2edge.append([len(out_edges)])
                    out_junctions.append(seg[:,i])
                    out_junctions_semantic.append(2 - int(mod[i]))
                    assert np.linalg.norm(j12_img_px[:,old_idx] - seg[:,i]) < CMP_EPS
            out_edges.append(edge)
            edge2l_idx.append(l_idx)
            out_edges_semantic.append(edge_sem)

    """
    Occluded junctions should only have one edge
    """
    for i, (jsem, edge_list) in enumerate(zip(out_junctions_semantic, junc2edge)):
        assert not (jsem == 1 and len(edge_list) > 1)


    # Now add negative lines
    if out_junctions:
        out = {}
        out_junctions = np.array(out_junctions)
        out_edges = np.array(out_edges)
        out['junctions'] = out_junctions.tolist()
        out['edges_positive'] = out_edges.tolist()
        out['junctions_semantic'] = out_junctions_semantic
        out['edges_semantic'] = [REDUCED_SINGLE_LINE_CLASSES_MAPPING[l] for l in out_edges_semantic]
        out['edges_all_semantic'] = out_edges_semantic
        out['edges_negative'] = generate_negative_edges(image, out_junctions, out_edges)
    else:
        out = None

    return edge2l_idx, out

def line_to_front(line_points_in):
    """ Project a line segment in 3D to be in front of camera, assuming line is in camera homegenous coordinates.
    I.e. Camera position is at origin.
    line_points: 3x2, each column being a point.
    """
    behind = line_points_in[2] < 0

    # Both end-points behind camera?
    if np.all(behind):
        return behind, None

    line_points = np.copy(line_points_in)

    if line_points[2,0] < 0:
        q = line_points[:,0] - line_points[:,1]
        line_points[:,0] = line_points[:,1] -(line_points[2,1]/(q[2]+DIV_EPS))*q
        line_points[2,0] = 0 # Make sure it is actually 0 since this assumed in occlusion check
    elif line_points[2,1] < 0:
        q = line_points[:,1] - line_points[:,0]
        line_points[:,1] = line_points[:,0] -(line_points[2,0]/(q[2]+DIV_EPS))*q
        line_points[2,1] = 0 # Make sure it is actually 0 since this assumed in occlusion check

    if np.any(line_points[2] < -CMP_EPS):
        print('line_points_in', line_points_in)
        print('line_points', line_points)

    return behind, line_points

def line_to_img(line_points, K, img_poly = None, width = None, height = None, l_idx = None):
    """ Project a line segment in 3D FOV in image pixels.  Assumes camera is at origin and line in front of camera.
    line_points: 3x2, each column being a point.
    """
    assert np.all(line_points[2] > -CMP_EPS)

    # Construct image polygon if not supplied
    if not img_poly:
        img_poly = sg.box(0, 0, width, height)

    # Line in Pixel coordinates
    line_points_px = K@line_points
    line_points_px /= (line_points_px[2] + DIV_EPS)
    line = sg.LineString(line_points_px[:2].T)

    # Line in image bounds
    try:
        line_img = img_poly.intersection(line)
    except shapely.errors.TopologicalError:
        # Shapely precision errors may be solved by using buffer to alter representation.
        line_img = img_poly.intersection(line.buffer(0))

    assert not isinstance(line_img, sg.Polygon)

    # Check if line was inside image bounds
    unmodified = np.zeros(2, dtype=np.bool)
    if line_img.is_empty or isinstance(line_img, sg.Point):
        new_line_points = None
    else:
        new_line_points = np.array(line_img.coords).T
        for i in range(2):
            unmodified |= (np.linalg.norm(line_points_px[:2] - new_line_points[:,i,None], axis=0) < 1e-5)
        # if l_idx:
        #     plt.figure()
        #     plt.plot(*np.array(img_poly.boundary.coords).T, 'b.-')
        #     plt.plot(*line_points_px[:2], 'r.-')
        #     plt.plot(*new_line_points, 'g.-')
        #     for i in range(2):
        #             # plt.text(*new_line_points[:,i], 'Diff: {}'.format(line_points_px[:2] - new_line_points[:,i]))
        #             plt.text(*new_line_points[:,i], 'Diff: {}'.format(np.linalg.norm(line_points_px[:2] - new_line_points[:,i,None], axis=0)), rotation=45)
        #             plt.plot(*new_line_points[:,i], 'b.' if unmodified[i] else 'b*')
        #     plt.title('Points: {}'.format(line_points_px[:2]))
        #     plt.savefig('/host_home/plots/hawp/debug/toline_{:03d}.svg'.format(l_idx))
        #     plt.close()

    return ~unmodified, new_line_points


def get_visible_segments(modified, line_points, plane_list, plane_line_mask, junctions, l_idx = None, ann_3D = None):
    """ Assumes camera is at origin, checks if line is visible due to planes.
    Assumes line in front of camera.
    line_points: 3x2, each column being a point.
    planes: List of Plane class.
    plane_line_mask: Bool vector of which planes the line belong to.
    """
    assert np.all(line_points[2] > -CMP_EPS)
    #Make all z positive
    line_points[2] = np.abs(line_points[2])

    match_idx_out = [DEF_IDX_MATCH]
    line_segments_out = [line_points.copy()]
    DBG_PLANE_IDX = []
    projected_lines = []

    # For each plane check if the line segment is occluded
    for p_idx, plane in enumerate(plane_list):

        if plane_line_mask[p_idx]:
            #Skip check if line is in plane
            continue
        # if l_idx == DBG_LIDX: print(p_idx, ann_3D['planes'][p_idx]['semantic'])

        if plane.semantic == 'outwall':
            continue

        valid_modified = []
        valid_segments = []
        valid_match_idx = []
        for line_points, match_idx in zip(line_segments_out, match_idx_out):
            new_segments, new_match_idx, proj_lines = add_visible_segments_single_plane(line_points, plane, P_IDX=p_idx, l_idx = l_idx)
            valid_segments += new_segments

            projected_lines += proj_lines
            remap_match = []
            for m_idx in new_match_idx:
                remap_match.append({new_idx:match_idx[old_idx] for new_idx, old_idx in m_idx.items()})
            valid_match_idx += remap_match
            if len(proj_lines) > 0 and l_idx == DBG_LIDX:
                DBG_PLANE_IDX.append(p_idx)
        # if l_idx == DBG_LIDX: print(len(valid_modified), len(valid_segments))

        match_idx_out = valid_match_idx
        line_segments_out = valid_segments

        # Stop if there are no valid segments left
        if not line_segments_out:
            break

    # Adjust modify based on match idx
    # print('l',len(valid_match_idx))
    #TODO: UnModified without match?
    modified_out = []
    for m_idx in match_idx_out:
        mod = np.zeros_like(modified)
        for new_idx in range(2):
            old_idx = m_idx[new_idx]
            mod[new_idx] = True if old_idx is None else modified[old_idx]
        modified_out.append(mod)

    #DEBUG: Visualize occluding and non-occluding planes
    if False and l_idx == DBG_LIDX:
        _, junction_pairs = np.array(ann_3D['lineJunctionMatrix']).nonzero()
        junction_pairs = junction_pairs.reshape(-1, 2)
        _, plane_lines = (np.array(ann_3D['planeLineMatrix'])[DBG_PLANE_IDX]).nonzero()
        # window_plane_idx = np.flatnonzero([p['semantic']=='balcony' for p in ann_3D['planes']])
        # window_lines = []
        # window_line_colors = []
        # for w_idx in window_plane_idx:
        #     w_lines = np.flatnonzero(np.array(ann_3D['planeLineMatrix'])[w_idx])
        #     window_lines += w_lines.tolist()
        #     window_line_colors += np.repeat(np.random.ranf([1,3]), w_lines.size, axis=0).tolist()

        print(DBG_PLANE_IDX) #26
        print([len(plane_list[idx]) for idx in DBG_PLANE_IDX])
        if DBG_PLANE_IDX:
            occluding_junctions = np.hstack([plane_list[idx].get_polygon_exterior() for idx in DBG_PLANE_IDX])
            print(occluding_junctions.shape)
            occluding_junction_colors = np.repeat(np.array([[0.0, 1.0, 0.0]]), occluding_junctions.shape[1], axis=0)
            occluding_junction_set = open3d.geometry.PointCloud()
            occluding_junction_set.points = open3d.utility.Vector3dVector(occluding_junctions.T)
            occluding_junction_set.colors = open3d.utility.Vector3dVector(occluding_junction_colors)
        else:
            occluding_junction_set = open3d.geometry.PointCloud()

        other_junctions = np.hstack([p.get_polygon_exterior() for (idx, p) in enumerate(plane_list) if idx not in DBG_PLANE_IDX])
        junction_colors = np.repeat(np.array([[1.0, 0.0, 0.0]]), other_junctions.shape[1], axis=0)
        junction_set = open3d.geometry.PointCloud()
        junction_set.points = open3d.utility.Vector3dVector(other_junctions.T)
        junction_set.colors = open3d.utility.Vector3dVector(junction_colors)


        origin_set = open3d.geometry.PointCloud()
        origin_set.points = open3d.utility.Vector3dVector(np.zeros([1,3]))
        origin_set.colors = open3d.utility.Vector3dVector(np.zeros([1,3]))
        origin_set.normals = open3d.utility.Vector3dVector(np.array([[0,0,1]]))

        line_colors = np.repeat(np.array([[1.0, 0.0, 0.0]]), junction_pairs.shape[0], axis=0)
        # line_colors[window_lines] = window_line_colors
        line_colors[plane_lines] = [0,0,1]
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(junctions.T)
        line_set.lines = open3d.utility.Vector2iVector(junction_pairs)
        line_set.colors = open3d.utility.Vector3dVector(line_colors)

        chosen_line_set = open3d.geometry.LineSet()
        chosen_junction_set = open3d.geometry.PointCloud()

        if line_segments_out:
            if len(line_segments_out) > 1:
                line_segments_np = np.hstack(line_segments_out)
            else:
                line_segments_np = line_segments_out[0]
            chosen_line_set.points = open3d.utility.Vector3dVector(line_segments_np.T)
            chosen_line_set.lines = open3d.utility.Vector2iVector(np.arange(line_segments_np.shape[1]).reshape([-1,2]))
            line_colors = np.repeat(np.array([[0.5,0.5, 0.0]]), line_segments_np.shape[1], axis=0)
            chosen_line_set.colors = open3d.utility.Vector3dVector(line_colors)

            chosen_junction_set.points = open3d.utility.Vector3dVector(line_segments_np.T)
            j_colors = np.repeat(np.array([[0.5,0.5, 0.0]]), line_segments_np.shape[1], axis=0)
            chosen_junction_set.colors = open3d.utility.Vector3dVector(j_colors)

        original_line_set = open3d.geometry.LineSet()
        original_line_set.points = open3d.utility.Vector3dVector(line_points.T)
        original_line_set.lines = open3d.utility.Vector2iVector(np.arange(2).reshape([-1,2]))
        line_colors = np.array([[0,0.5, 0.5]])
        original_line_set.colors = open3d.utility.Vector3dVector(line_colors)

        proj_line_set = open3d.geometry.LineSet()
        if projected_lines:
            projected_lines = np.hstack(projected_lines)
            proj_line_set.points = open3d.utility.Vector3dVector(projected_lines.T)
            proj_line_set.lines = open3d.utility.Vector2iVector(np.arange(projected_lines.shape[1]).reshape([-1,2]))
            np.repeat(np.array([[0.1,0.1, 0.0]]), projected_lines.shape[1], axis=0)
            proj_line_set.colors = open3d.utility.Vector3dVector(line_colors)


        open3d.visualization.draw_geometries([origin_set, chosen_junction_set, chosen_line_set, occluding_junction_set, junction_set, line_set, original_line_set, proj_line_set])

    return line_segments_out, modified_out, match_idx_out

DBG_LIDX = -1
# DBG_LIDX = 43
def add_visible_segments_single_plane(line_points, plane, P_IDX = 0, l_idx = None):
    '''
    Adds visible line segments to valid_segments. Returns True if occlusion found.
    '''

    valid_segments = [] # All visible line segments.
    valid_match_idx = [] # 0 or 1 to indicate if the point matches the line_points received. None for new "fake" junction.

    #Figure out where the viewing ray cuts the plane
    # 0 < dist_frac < 1 => plane between camera and point
    p_params = plane.params
    dist_frac = - p_params[3]/(p_params[:3]@line_points + DIV_EPS)
    occluded = (0 < dist_frac) & (dist_frac < 1)
    in_plane = np.isclose(dist_frac, 1)

    occluded[in_plane] = False

    #TODO: Handle the case of point being on the plane as a special case.
    # For example there is no need to compute free segment if we have the free point not occluded
    #Make sure dist frac is greater than one if we consider the point free.
    # frac_mask = (~occluded) & (dist_frac>0)
    # dist_frac[frac_mask] = np.maximum(1,dist_frac[frac_mask])

    # Projection to plane along viewing ray.
    endp_plane = dist_frac*line_points

    if l_idx == DBG_LIDX: print('--------------Pidx: ', P_IDX, 'Frac:', dist_frac, 'Occluded: ', occluded, '------------------')
    if l_idx == DBG_LIDX: print('endp_plane',endp_plane)
    if l_idx == DBG_LIDX: print('in_plane',in_plane)

    if not np.any(occluded):
    # if True:
        # No occlusion, skip
        valid_segments.append(line_points)
        valid_match_idx.append(DEF_IDX_MATCH)
        if l_idx == DBG_LIDX: print('OK')
        return valid_segments, valid_match_idx, []


    # Compute line direction and start point.
    line_v = line_points[:,1]-line_points[:,0]
    l0 = line_points[:,0]

    if ~np.all(occluded | in_plane):
        """
        If one of the points are in front of the plane we need to find the intersection
        between line and plane and study the occluded segment
        """
        free_idx = np.flatnonzero(~occluded)[0]
        mod_idx = (free_idx + 1) % 2
        line_dist = -(p_params[:3]@l0 + p_params[3])/(p_params[:3]@line_v + DIV_EPS)
        line_length = np.linalg.norm(line_v)
        free_to_mod_dist = line_dist if free_idx == 0 else 1-line_dist
        if l_idx == DBG_LIDX: print('free_to_mod_dist', free_to_mod_dist)
        if l_idx == DBG_LIDX: print('free_to_mod_dist_mm', np.linalg.norm(line_v)*free_to_mod_dist)

        if free_to_mod_dist*line_length < CMP_EPS:
            # No part of the line is free
            free_idx = None
            mod_idx = None
            if l_idx == DBG_LIDX: print('No part free')
        elif line_length*np.abs(free_to_mod_dist - 1) < CMP_EPS:
            # All line is free
            valid_segments.append(line_points)
            valid_match_idx.append(DEF_IDX_MATCH)
            if l_idx == DBG_LIDX: print('OK2')
            return valid_segments, valid_match_idx, []
        else:
            #Modify endpoint
            endp_plane[:,free_idx] = l0 + line_dist*line_v

        if not (((-CMP_EPS < line_dist) and (line_dist < 1 + CMP_EPS))):
            print('Frac:', dist_frac, 'Occluded: ', occluded)
            print(line_dist)
            print('free, mod', free_idx, mod_idx)
            print('line_v', line_v)
            print('l0', l0)
        assert ((-CMP_EPS < line_dist) and (line_dist < 1 + CMP_EPS))
        if l_idx == DBG_LIDX: print('One occluded')
    else:
        free_idx = None
        mod_idx = None
        line_dist = 0
        if l_idx == DBG_LIDX: print('All occluded')

    if l_idx == DBG_LIDX:
        camera_c = np.zeros([3,1])
        c_to_lp0 = np.hstack([camera_c,line_points[:,0,None]])
        c_to_lp1 = np.hstack([camera_c,line_points[:,1,None]])
        plot_cfgs = [
            {'points': endp_plane,      'options': 'g.-'},
            {'points': plane.get_exterior(), 'options': 'r.-'},
            {'points': camera_c,        'options': 'ko'},
            {'points': line_points,     'options': 'b.-'},
            {'points': c_to_lp0,        'options': 'b--'},
            {'points': c_to_lp1,        'options': 'b--'},
        ]

        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        line_length = np.linalg.norm(line_points[:,0]-line_points[:,1])
        ax.set_title('df {}, ld {:.3g}, ll {:.3g}'.format(dist_frac, line_dist, line_length))
        for plt_c in plot_cfgs:
            ax.plot(*plt_c['points'], plt_c['options'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax = fig.add_subplot(222)
        ax.set_title('XY')
        for plt_c in plot_cfgs:
            ax.plot(*plt_c['points'][:2], plt_c['options'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax = fig.add_subplot(223)
        ax.set_title('YZ')
        for plt_c in plot_cfgs:
            ax.plot(*plt_c['points'][1:], plt_c['options'])
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax = fig.add_subplot(224)
        ax.set_title('ZX')
        for plt_c in plot_cfgs:
            ax.plot(*plt_c['points'][[2,0]], plt_c['options'])
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        plt.tight_layout()

        plt.savefig('/host_home/plots/hawp/debug/3D_{:03d}.svg'.format(P_IDX))
        plt.close(fig)


    """
    Project point on plane and take intersection
    """
    disjoint, visible_segments = plane.difference(endp_plane)


    if disjoint:
        """ No occlusion, line and plane are not overlapping """
        valid_segments.append(line_points)
        valid_match_idx.append(DEF_IDX_MATCH)
        assert np.linalg.norm(line_points[:,0] - line_points[:,1]) > 1e-5
        if l_idx == DBG_LIDX: print('No overlap')
        return valid_segments, valid_match_idx, []


    free_segment_connected = False
    nbr_appended = 0
    if free_idx is None:
        free_segment = None
    else:
        free_segment = line_points.copy()
        free_segment[:,mod_idx] = endp_plane[:,free_idx]

    # if l_idx == DBG_LIDX:
    #     fig = plt.figure()

    if l_idx == DBG_LIDX:
        plane.plot([endp_plane])
        plt.savefig('/host_home/plots/hawp/debug/full_line_{}_p{}.svg'.format(l_idx, P_IDX))
        plt.close()

        plane.plot(visible_segments)
        plt.savefig('/host_home/plots/hawp/debug/visible_segments_l{}_p{}.svg'.format(l_idx, P_IDX))
        plt.close()

    for plane_seg in visible_segments:

        """
        Now we find the point on the line which projects to this point in the plane.
        Solve x1*p - x2*l = l0
        Where p are segment points projected to the plane and x2*l + l0 are all points on the line.
        """
        new_line_points = np.zeros_like(line_points)
        for i in range(2):
            A = np.hstack([plane_seg[:2,i,None], -line_v[:2, None]])
            x = np.linalg.solve(A, l0[:2])
            # A = np.hstack([plane_seg[:,i,None], -line_v[:, None]])
            # x = np.linalg.lstsq(A, l0,rcond=None)
            assert -CMP_EPS < x[1] < 1 + CMP_EPS
            if l_idx == DBG_LIDX: print('x{}'.format(i), x)
            new_line_points[:,i] = x[0]*plane_seg[:,i]

        if l_idx == DBG_LIDX: print('plane seg')
        if l_idx == DBG_LIDX: print(plane_seg)

        """ See if this segment can be appended to the free segment """
        if free_segment is not None:
            for i in range(2):
                appendable_idx = np.flatnonzero(np.linalg.norm(free_segment[:,i,None] - new_line_points, axis=0) < CMP_EPS)
                if l_idx == DBG_LIDX: print('idx: {}, Dist free segment {}'.format(appendable_idx, np.linalg.norm(free_segment[:,i,None] - new_line_points, axis=0)))
                if len(appendable_idx) > 0:
                    new_line_points[:,appendable_idx] = free_segment[:,(i+1)%2, None]
                    free_segment_connected = True
                    if l_idx == DBG_LIDX: print('Appending to free segment')
                    break

        # Check that the segment is long enough
        # if np.linalg.norm(new_line_points[:,0] - new_line_points[:,1]) < MIN_LEN:
        #     continue

        # Check if any end points are original points, we have lost the order.
        match_idx = {k:None for k in DEF_IDX_MATCH}
        pair_dist2 = cdist(new_line_points.T, line_points.T, 'sqeuclidean')
        for new_idx, old_idx in zip(*np.nonzero(pair_dist2 < CMP_EPS2)):
            match_idx[new_idx] = old_idx

        valid_segments.append(new_line_points)
        valid_match_idx.append(match_idx)
        nbr_appended += 1
        if l_idx == DBG_LIDX:
             print('Append part')
             print('plane params',plane.params)
             print('New')
             print(new_line_points)
             print('Old')
             print(line_points)
             # plt.plot(*np.array(line_segment.coords).T, 'g.-')

    # if l_idx == DBG_LIDX:
    #     if nbr_appended>0:
            # fig = plane.plot()
            # plt.savefig('/host_home/plots/hawp/debug/adjusted{:03d}.svg'.format(P_IDX))
        # plt.close(fig)

    ''' The free segment could not be connected to another segment.'''
    if not free_segment_connected and free_idx is not None:
        new_line_points = line_points.copy()
        new_line_points[:,mod_idx] = endp_plane[:,free_idx]
        if np.linalg.norm(new_line_points[:,0] - new_line_points[:,1]) > MIN_LEN:
            not_mod_idx = int(not mod_idx)
            match_idx = {k:None for k in DEF_IDX_MATCH}
            match_idx[not_mod_idx] = not_mod_idx
            valid_segments.append(new_line_points)
            valid_match_idx.append(match_idx)
            if l_idx == DBG_LIDX: print('Free segment on its own')
        else:
            if l_idx == DBG_LIDX: print('Free segment to small')


    return valid_segments, valid_match_idx, [endp_plane]

def generate_negative_edges(image, junctions, pos_edges):
    heatmap_scale = (128,128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    lmap = np.zeros(heatmap_scale, dtype=np.float32)

    scaled_junctions = np.clip(junctions*np.array([[fx,fy]]), 0, heatmap_scale[0]-1e-1)
    lineset = set()
    lneg = []

    for l_edge in pos_edges:
        lineset.add(frozenset(l_edge))
        e1,e2 = l_edge
        jpos_int = scaled_junctions[l_edge,:].astype(np.int)
        rr, cc, value = skimage.draw.line_aa(*jpos_int[0,::-1], *jpos_int[1,::-1])
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    llmap = scipy.ndimage.zoom(lmap, [0.5, 0.5])
    for l_edge in combinations(range(scaled_junctions.shape[0]), 2):
        if frozenset(l_edge) not in lineset:
            jpos_int = scaled_junctions[l_edge,:].astype(np.int)//2
            rr, cc, value = skimage.draw.line_aa(*jpos_int[0,::-1], *jpos_int[1,::-1])
            lneg.append([l_edge, np.average(np.minimum(value, llmap[rr, cc]))])

    if len(lneg) == 0:
        return []

    lneg.sort(key=lambda l: -l[-1])

    Lneg = [l[0] for l in lneg][:NBR_NEG_LINES]
    return Lneg



def generate_sub_ann(data_dir, scene_dir, out_image_dir, out_json_dir, make_plot=False, invalid_rooms = None):
    json_path = osp.join(out_json_dir, scene_dir)
    if not osp.exists(json_path) or broken_symlink(out_image_dir, json_path):
        generated = True
        ann = link_and_annotate(data_dir, scene_dir, out_image_dir, make_plot=make_plot, invalid_rooms = invalid_rooms)
        with open(json_path, 'w') as f:
            json.dump(ann, f)
    else:
        generated = False

    return generated

def broken_symlink(out_image_dir, json_path):
    with open(json_path, 'r') as f:
        ann = json.load(f)

    for a in ann:
        try:
            os.stat(osp.join(out_image_dir, a['filename']))
        except OSError as e:
            if e.errno == errno.ENOENT:
                return True
    return False

def merge_json(sub_json_dir, final_json_dir, mini_size = {'train':4000, 'test':1000, 'val': 1000}):
    ann = {'train':[],'val':[],'test':[]}
    nbr_scenes = {key:0 for key in ann}

    for json_file in os.listdir(sub_json_dir):
        scene_id = int(json_file.split('_')[1])
        with open(osp.join(sub_json_dir, json_file), 'r') as f:
            scene_ann = json.load(f)
        for dset, dlist in ann.items():
            r = DATA_RANGE[dset]
            if r[0] <= scene_id < r[1]:
                dlist += scene_ann
                nbr_scenes[dset] += 1
                break

    for dset, dlist in ann.items():
        with open(osp.join(final_json_dir, '{}.json'.format(dset)), 'w') as f:
            json.dump(dlist, f)

        if len(dlist) > mini_size[dset]:
            mini_ann = random.sample(dlist, mini_size[dset])
            with open(osp.join(final_json_dir, '{}_mini.json'.format(dset)), 'w') as f:
                json.dump(mini_ann, f)

    nbr_images = {dset:len(dlist) for dset,dlist in ann.items()}
    all_ann = ann['train'] + ann['test'] + ann['val']

    return all_ann, nbr_scenes, nbr_images


class InvalidRooms:
    def __init__(self, txt_path):
        self.invalid_scenes = set()
        self.invalid_rooms = {}

        prog = re.compile(r'^scene_(?P<sID>\d+)(?:_room_(?P<rID>\d+))*\w*$')
        # room_p = re.complie(r'^room_(?Pid\d+)')
        with open(txt_path, 'r') as f:
            for l in f.readlines():
                match = prog.match(l)
                if match:
                    if match.group('rID'):
                        self._add_room(match.group('sID'), match.group('rID'))
                    else:
                        self._add_scene(match.group('sID'))


    def _add_room(self, scene, room):
        if scene in self.invalid_rooms:
            self.invalid_rooms[scene].add(room)
        else:
            self.invalid_rooms[scene] = {room}

    def _add_scene(self, scene):
        self.invalid_scenes.add(scene)

    def room_is_invalid(self, scene, room):
        try:
            invalid_room  = room in self.invalid_rooms[scene]
        except KeyError:
            invalid_room = False

        return self.scene_is_invalid(scene) or invalid_room

    def scene_is_invalid(self, scene):
        return scene in self.invalid_scenes

def autolabel_bar(ax ,rects, bar_label):
    for idx,rect in enumerate(rects):
        height = rect.get_height()
        # ax.text(rect.get_x() , 1.05*height,
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=90)

def compute_label_stats(line_c, labels, ax = None):
    hist, bin_edges = np.histogram(line_c, bins=range(len(labels)+1))

    if ax:
        bar_plot = ax.bar(bin_edges[:-1], hist, align='center')
        autolabel_bar(ax, bar_plot, hist)
        ax.set_title('line labels')
        ax.set_ylabel('Nbr Lines')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(['-'.join(c) for c in labels], rotation=45, ha='right')
        ax.set_ylim([0, np.max(hist)*1.5])


    #Assume balance between positive and negative examples during training.
    total_instances = np.sum(hist)
    nbr_negative_edges = total_instances
    fake_total = total_instances + nbr_negative_edges
    fake_hist = np.copy(hist)

    fake_hist[0] = nbr_negative_edges
    valid_mask = (fake_hist != 0)
    weights = np.zeros(len(labels))
    weights[valid_mask] = (1.0/fake_hist[valid_mask])
    weights /= np.sum(weights)
    bias = fake_hist/fake_total
    # YAML stats
    if isinstance(labels[0],frozenset):
        labels = ['-'.join(c) for c in labels]
    stats = {
        'class_names': labels,
        'nbr_line_classes': [int(N) for N in fake_hist],
        'weight_line_classes': [float(N) for N in weights],
        'bias_line_classes': [float(N) for N in bias],
    }
    for i in range(len(labels)):
        print('{} - N:{}, W:{:.02g}, B:{:.02g}'.format(
            stats['class_names'][i],
            stats['nbr_line_classes'][i],
            stats['weight_line_classes'][i],
            stats['bias_line_classes'][i],
        ))

    return stats


if __name__ == '__main__':
    script_path = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(description='Generate wireframe format from Structured3D', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str, help='Path to Structured3D')
    parser.add_argument('out_dir', type=str, help='Path to storing conversion')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')
    parser.add_argument('-s', '--nbr-scenes', type=int, default = None, help='Number of scenes to process')
    parser.add_argument('-l', '--logfile', type=str, default = None, help='logfile path if wanted')
    parser.add_argument('--invalid', type=str, help='Invalid list from Structured3D',
                        default = osp.abspath(osp.join(script_path, '..', 'libs', 'Structured3D', 'metadata', 'errata.txt')))
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite out_dir if existing')
    parser.add_argument('--halt', action='store_true', help='Halt on error')
    parser.add_argument('--merge-only', action='store_true', help='Only do merge, skip looking for new annotations')
    parser.add_argument('--profile', action='store_true', help='Run profiler on one scene')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot images with GT lines')

    args = parser.parse_args()

    # create logger
    logger = logging.getLogger('structured3D2wireframe')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args.logfile:
        fh = logging.FileHandler(args.logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    invalid_rooms = InvalidRooms(args.invalid) if args.invalid else None

    if osp.exists(args.out_dir) and not args.overwrite:
        print("Output directory {} already exists, specify -o flag if overwrite is permitted".format(args.out_dir))
        sys.exit()

    out_image_dir = osp.join(args.out_dir, 'images')
    os.makedirs(out_image_dir, exist_ok = True)
    out_json_dir = osp.join(args.out_dir, 'ann')
    os.makedirs(out_json_dir, exist_ok = True)

    dirs = (os.listdir(args.data_dir))

    if args.profile:
        pr= cProfile.Profile()
        pr.enable()
        link_and_annotate(args.data_dir, dirs[0], out_image_dir, make_plot=False)
        pr.disable()
        with open('stats.profile', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_stats()
        sys.exit()

    if args.nbr_scenes:
        dirs = dirs[:args.nbr_scenes]

    if args.merge_only:
        dirs = []

    result = []
    start = time.time()
    nbr_failed = 0
    nbr_invalid = 0
    nbr_generated = 0
    nbr_existed = 0

    with mp.Pool(processes=args.nbr_workers) as pool:
        for scene_dir in dirs:
            # if not scene_dir.endswith('00173'):
            #     continue
            f_args = (args.data_dir, scene_dir, out_image_dir, out_json_dir)
            f_kwargs = dict(make_plot=args.plot,
                            invalid_rooms = invalid_rooms)
            r = pool.apply_async(generate_sub_ann, f_args, f_kwargs)
            result.append(r)

        #Wait for results, waits for processes to finish and raises errors
        for i, r in enumerate(tqdm(result)):
            try:
                generated = r.get()
                nbr_generated += generated
                nbr_existed += not generated
            except KeyboardInterrupt:
                raise
            except InvalidGeometryError:
                logger.exception('Invalid geometry for scene {}'.format(dirs[i]))
                nbr_invalid += 1
            except:
                logger.exception('Got exception for scene {}'.format(dirs[i]))
                nbr_failed += 1
                if args.halt:
                    raise

    logger.info('Generating {} scenes took {}'.format(len(dirs), timedelta(seconds=time.time()-start)))
    logger.info('{}/{} scenes generated'.format(nbr_generated, len(dirs)))
    logger.info('{}/{} scenes existed'.format(nbr_existed, len(dirs)))
    logger.info('{}/{} scenes failed'.format(nbr_failed, len(dirs)))
    logger.info('{}/{} scenes invalid'.format(nbr_invalid, len(dirs)))


    start = time.time()
    ann, nbr_scenes_split, nbr_images_split = merge_json(out_json_dir, args.out_dir)
    nbr_scenes_merged = np.sum([v for k,v in nbr_scenes_split.items()])
    logger.info('Merging {} scenes took {}'.format(nbr_scenes_merged,  timedelta(seconds=time.time()-start)))
    logger.info('Scene split is:')
    for k,v in nbr_scenes_split.items():
        print(k,':',v)

    nbr_images = len(ann)
    r = {}
    r['nbr_junctions'] = np.array([len(a['junctions']) for a in ann])
    r['nbr_visible_junctions'] = np.array([(np.array(a['junctions_semantic'], dtype=np.int) == 1).sum() for a in ann])
    r['nbr_edges_pos'] = np.array([len(a['edges_positive']) for a in ann])
    r['nbr_edges_neg'] = np.array([len(a['edges_negative']) for a in ann])

    fig, ax = plt.subplots(2,2)
    for i,title in enumerate(r):
        ax1 = ax.flat[i]
        ax1.hist(r[title], bins=15)
        ax1.set_title(title)
        ax1.set_ylabel('Nbr images (total: {})'.format(nbr_images))
        ax1.set_xlabel('{} / image'.format(title))

    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, 'stats.svg'))
    plt.close()


    line_c = np.concatenate([np.array(a['edges_all_semantic'], dtype=np.int32) for a in ann])

    fig, ax = plt.subplots(2,1)
    stats_all_labels = compute_label_stats(line_c, ALL_SINGLE_LINE_CLASSES, ax=ax[0])
    stats_all_labels['nbr_images_split'] = nbr_images_split
    stats_all_labels['nbr_scenes_split'] = nbr_scenes_split

    with open(osp.join(args.out_dir, 'stats_all.yaml'), 'w') as f:
        yaml.safe_dump(stats_all_labels, f, default_flow_style=None)

    line_c = np.concatenate([np.array(a['edges_semantic'], dtype=np.int32) for a in ann])
    stats_simple_labels = compute_label_stats(line_c, REDUCED_SINGLE_LINE_CLASSES, ax=ax[1])

    with open(osp.join(args.out_dir, 'stats_simple.yaml'), 'w') as f:
        yaml.safe_dump(stats_simple_labels, f, default_flow_style=None)

    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, 'label_stats.svg'))
    plt.close()
