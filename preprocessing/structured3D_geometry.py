import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
from shapely.errors import TopologicalError
from copy import deepcopy
from descartes import PolygonPatch
import seaborn as sns
import os
import os.path as osp
import cv2
import csv


CMP_EPS = 1e-2
CMP_EPS2 = CMP_EPS**2
MIN_LEN = 10 # Minimul line lenght in mm
MAX_PLANE_DIST = 1 # Maximum deviation between plane params and junction coordinates
DIV_EPS = 1e-15
door_rgb = np.array([214, 39, 40], dtype=np.uint8)
# Many doors are behind curtains, so we consider these closed as well.
curtain_rgb = np.array([219, 219, 141], dtype=np.uint8)

class InvalidGeometryError(Exception):
    pass
def normalize(vector):
    return vector / (np.linalg.norm(vector) + DIV_EPS)

def parse_camera_info(camera_info, width, height):
    """ extract intrinsic and extrinsic matrix
    Make K, R and t s.t. lambda*x = K*(R*X + t)
    where lambda is any scalar, x is point in image in pixels and X is point in world coordinates
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = np.cross(W, U)

    R = np.vstack((U, V, W))

    camera_pos = np.array(camera_info[:3]).reshape(3,1)

    t = -R@camera_pos

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return R, t, K

class Plane:
    """
    # Creates plane polygons using shapely from Structured3D data
    # Will treat wall planes with door and window lines differently to account for holes in the geometry.
    # Will store inplane coordinates and transforms
    """
    def __init__(self, ann, line_junction_matrix, line_is_door_mask, line_is_window_mask, junctions, open_doors = True):
        self.id = ann['ID']
        self.params = np.array(ann['normal'] + [ann['offset']], dtype=np.float32)#.reshape([4,1])
        self.params /= np.linalg.norm(self.params[:3]) #Normalize normal vector
        self.semantic = ann['semantic']
        self.type = ann['type']


        # Validate geometry
        plane_junctions = junctions[:, np.any(line_junction_matrix, axis=0)]
        nbr_junctions = plane_junctions.shape[1]
        if nbr_junctions < 3:
            raise InvalidGeometryError('There are only {} plane junctions in this plane.'.format(nbr_junctions))

        nbr_lines = line_junction_matrix.shape[0]
        if nbr_lines < 3:
            raise InvalidGeometryError('There are only {} lines in this plane.'.format(nbr_lines))

        dist = np.max(np.abs(self.params[:3]@plane_junctions + self.params[3]))
        if dist > MAX_PLANE_DIST:
            self.params = self._estimate_plane_params(plane_junctions)
            dist = np.max(np.abs(self.params[:3]@plane_junctions + self.params[3]))
            if dist > MAX_PLANE_DIST:
                raise InvalidGeometryError('Plane junctions does not satisfy plane parameter constraints. {} > {}'.format(dist, MAX_PLANE_DIST))


        # Construct polygons
        self._make_transform(plane_junctions)
        self._gen_polygon(line_junction_matrix, line_is_door_mask, line_is_window_mask, junctions, open_doors)


    def _estimate_plane_params(self, junctions):
        # 3xN matrix of junctions
        nbr_junctions = junctions.shape[1]
        assert nbr_junctions > 2
        junctions_h = np.vstack([junctions, np.ones(nbr_junctions).reshape([1,-1])])

        # Normalize
        N = np.eye(4)
        j_mean = np.mean(junctions, axis=1)
        j_std = np.std(junctions-j_mean[:,None],axis=1) + DIV_EPS
        N[:3,3] = -j_mean
        N[:3] /= j_std.reshape([3,1])
        M = N@junctions_h

        # Minimize
        U,S,V = np.linalg.svd(M.T)
        params = N.T@V[-1]
        params /= np.linalg.norm(params[:3])
        return params.astype(np.float32)

    def _make_polygons(self, sub_line_junction_matrix, junctions, polygons = None, allow_convex_hull = False):
        if polygons is None:
            polygons = []
        nbr_lines = sub_line_junction_matrix.shape[0]
        if nbr_lines == 0:
            return polygons

        unq_junc_idx = np.flatnonzero(np.any(sub_line_junction_matrix, axis=0))
        line2junc_idx = [set(np.flatnonzero(row)) for row in sub_line_junction_matrix]
        junc2line_idx = {uj: set(np.flatnonzero(sub_line_junction_matrix[:,uj])) for uj in unq_junc_idx}

        ordered_junc_idx = list(line2junc_idx[0])
        traversed_lines = set([0])
        while ordered_junc_idx[0] != ordered_junc_idx[-1] and len(ordered_junc_idx) - 1 < nbr_lines:
            last_junc = ordered_junc_idx[-1]
            next_line = junc2line_idx[last_junc].difference(traversed_lines).pop()
            traversed_lines.add(next_line)
            next_junc = line2junc_idx[next_line].difference(set([last_junc])).pop()
            ordered_junc_idx.append(next_junc)

        inplane_junc = self.Tf_w2p(junctions[:,ordered_junc_idx])
        sg_poly = sg.Polygon(inplane_junc.T)
        if not sg_poly.is_valid and allow_convex_hull:
            points = sg.MultiPoint(sg_poly.boundary.coords[1:])
            sg_poly = points.convex_hull
        assert sg_poly.is_valid
        polygons.append(sg_poly)

        trav_mask = np.array([idx in traversed_lines for idx in range(nbr_lines)], dtype=np.bool)
        return self._make_polygons(sub_line_junction_matrix[~trav_mask], junctions, polygons = polygons, allow_convex_hull = allow_convex_hull)

    def _make_transform(self, polygon):
        """
        Align X axis with line and Z axis with plane normal
        Move origo to first point
        """
        W = normalize(self.params[:3])
        target_U = polygon[:,0] - polygon[:,1]
        orth_U = W.dot(target_U)*target_U
        U = normalize(target_U - orth_U)
        V = np.cross(W,U)
        R = R = np.vstack([U,V,W])
        t = t = -R@polygon[:,0].reshape([3,1])
        self.T = np.block([
            [R, t],
            [np.array([0,0,0,1])]
        ])

        self.Tf_w2p = lambda x: R@x + t # To inplane
        self.Tf_p2w = lambda x: R.T@(x - t) # To world


    def _cut_polygon(self, polygon, hole_polygons, ax= None):

        for hp in hole_polygons:
            polygon = polygon.difference(hp)

        assert isinstance(polygon, (sg.Polygon, sg.MultiPolygon))
        if ax:
            pstyle = ['--', 'dotted', '-.']
            g_attr = ['exterior', 'boundary', 'interiors']
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            poly_geoms = [polygon] if isinstance(polygon, sg.Polygon) else polygon.geoms
            for idx, ga, in enumerate(g_attr):
                ax = plt.subplot(2,3,4 + idx)
                ax.set_title(ga)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                for p in poly_geoms:
                    ax.add_patch(PolygonPatch(polygon, fc = sns.xkcd_rgb['blue'], ec = sns.xkcd_rgb['blue'], alpha = 0.5, zorder=1))
                    geom = getattr(p, ga, None)
                    if geom is None:
                        continue
                    try:
                        iter(geom)
                    except TypeError:
                        geom = [geom]

                    for idx, g in enumerate(geom):
                        ax.plot(*g.xy, linestyle=pstyle[idx%len(pstyle)], marker='o')


        return polygon

    def _gen_polygon(self, line_junction_matrix, line_is_door_mask, line_is_window_mask, junctions, open_doors=False):
        if not line_junction_matrix.shape[0] > 2 or not np.sum(np.any(line_junction_matrix, axis=0)) > 2:
            plt.figure()
            ax = plt.subplot(2,3,1)
            ax.set_title('All lines')
            for idx, line_mask in enumerate(line_junction_matrix):
                junc = self.Tf_w2p(junctions[:,line_mask])
                ax.plot(*junc[:2], 'b.-')
                ax.text(*np.mean(junc[:2], axis=1), idx)
            print('idx', self.id, 'junctions', junctions[:,np.any(line_junction_matrix, axis=0)])
            plt.tight_layout()
            plt.savefig('/host_home/plots/hawp/debug/cls_inplane_{}_poly.svg'.format(self.id))

            plt.close()

        assert line_junction_matrix.shape[0] > 2
        assert np.sum(np.any(line_junction_matrix, axis=0)) > 2
        if self.type != 'wall' or self.semantic == 'door' or self.semantic == 'window':

            allow_convex_hull = self.type in set(['ceiling', 'floor'])
            tmp_polygons = self._make_polygons(line_junction_matrix, junctions, allow_convex_hull=allow_convex_hull)
            self.hole_polygons = []
            self.closed_polygon = self.polygon = tmp_polygons[0] if len(tmp_polygons) == 1 else so.unary_union(tmp_polygons)
        else:
            normal_mask = ~line_is_door_mask & ~line_is_window_mask
            assert np.sum(normal_mask) > 3
            wall_polygons = self._make_polygons(line_junction_matrix[normal_mask], junctions)
            assert len(wall_polygons) == 1
            self.closed_polygon = wall_polygons[0]


            self.hole_polygons = self._make_polygons(line_junction_matrix[line_is_door_mask], junctions)

            if 0 < np.sum(line_is_door_mask) and np.sum(line_is_door_mask) < 4 :
                plt.figure()
                ax = plt.subplot(2,3,1)
                ax.set_title('All lines')
                for idx, line_mask in enumerate(line_junction_matrix):
                    junc = self.Tf_w2p(junctions[:,line_mask])
                    ax.plot(*junc[:2], 'b.-')
                    ax.text(*np.mean(junc[:2], axis=1), idx)

                ax = plt.subplot(2,3,2)
                ax.set_title('Main and Door')
                plt.plot(*self.closed_polygon.boundary.xy, 'r--', marker='o')
                ax.add_patch(PolygonPatch(self.closed_polygon, fc = sns.xkcd_rgb['red'], ec = sns.xkcd_rgb['red'], alpha = 0.5, zorder=1))

                for dp in self.hole_polygons:
                    plt.plot(*dp.boundary.xy, 'g--', marker='d')
                    ax.add_patch(PolygonPatch(dp, fc = sns.xkcd_rgb['green'], ec = sns.xkcd_rgb['green'], alpha = 0.5, zorder=2))

                plt.tight_layout()
                plt.savefig('/host_home/plots/hawp/debug/cls_inplane_{}_poly.svg'.format(self.id))

                plt.close()

            assert np.sum(line_is_door_mask) == 0 or np.sum(line_is_door_mask) > 3
            if self.hole_polygons and open_doors:
                # try:
                # self.polygon = self._cut_polygon(self.closed_polygon, self.hole_polygons, ax=ax)
                self.polygon = self._cut_polygon(self.closed_polygon, self.hole_polygons)
                self.open_idx = range(len(self.hole_polygons))
                # except:
                #     plt.savefig('/host_home/plots/hawp/debug/cls_inplane_{}_poly.svg'.format(self.id))
                #     plt.close()
                #     raise
            else:
                self.polygon = self.closed_polygon
            #
            # plt.tight_layout()
            # plt.savefig('/host_home/plots/hawp/debug/cls_inplane_{}_poly.svg'.format(self.id))
            #
            # plt.close()

    def _gen_door_sample_points(self, N = 10, margin = 20):
        # used when checking the semantic segmentation
        # Margin in mm
        samples = []
        for p in self.hole_polygons:
            minx, miny, maxx, maxy = p.bounds
            xv, yv = np.meshgrid(
                np.linspace(minx + margin, maxx - margin, N),
                np.linspace(miny + margin, maxy - margin, N),
                indexing='ij')
            sample_p = np.vstack([xv.ravel(),yv.ravel(), np.zeros(N*N)])
            sample_w = self.Tf_p2w(sample_p)
            assert np.all(np.abs(sample_p - self.Tf_w2p(sample_w)) < CMP_EPS)
            samples.append(sample_w)

        return samples

    def _set_door_open(self, open_idx):
        # used after checking the semantic segmentation
        self.open_idx = open_idx
        if open_idx:
            hole_polygons = [self.hole_polygons[idx] for idx in open_idx]
            self.polygon = self._cut_polygon(self.closed_polygon, hole_polygons)
        else:
            self.polygon = self.closed_polygon

    def get_polygon_holes(self):
        return [self.Tf_p2w(np.array(p.boundary.coords).T) for p in self.hole_polygons]

    def transform(self, T):
        # T: 4x4 matrix [R t ; 0 1] s.t. Wo = T*Wn.
        # Where Wo is current system and Wn is new coordinate system.
        # Returns a new object with the new coordinate system.
        assert np.abs(T[3,3]-1) < DIV_EPS

        cls = deepcopy(self)
        cls.T = cls.T@T
        R = cls.T[:3,:3]
        t = cls.T[:3,3,None]
        assert np.abs(np.linalg.det(R) -1) < CMP_EPS
        assert np.all(np.abs(R@R.T - np.identity(3)) < CMP_EPS)
        cls.Tf_w2p = lambda x: R@x + t # To inplane
        cls.Tf_p2w = lambda x: R.T@(x - t) # To world

        cls.params = T.T@cls.params
        cls.params /= np.linalg.norm(cls.params[:3])

        return cls

    def difference(self, line):
        # line: 2 endpoints as a 3x2 matrix
        # Assumes points are projected to the plane
        assert np.all(np.abs(self.params[:3]@line + self.params[3]) < CMP_EPS)

        line_sg = sg.LineString(self.Tf_w2p(line).T)

        if line_sg.disjoint(self.polygon):
            """ No occlusion, line and plane are not overlapping """
            return True, [line]

        visible_segments = line_sg.difference(self.polygon)
        if isinstance(visible_segments, sg.LineString):
            # There is only one segment
            visible_segments = [visible_segments]

        world_segments = []
        for v in visible_segments:
            if not v.is_empty and v.length > MIN_LEN:
                w_coords = self.Tf_p2w(np.array(v.coords).T)
                world_segments.append(w_coords)

        return False, world_segments

    def contains(self, points):
        # points: N endpoints as a 3xN matrix
        # Assumes points are projectede to the plane
        assert np.all(np.abs(self.params[:3]@points + self.params[3]) < CMP_EPS)

        points_p = self.Tf_w2p(points)
        cmask = np.array([
            self.polygon.contains(sg.Point(p)) for p in points_p.T
            ])
        return cmask

    def get_exterior(self):
        return self.Tf_p2w(np.array(self.polygon.exterior.coords).T)

    def plot(self, lines):

        fig = plt.figure()
        ax = plt.gca()
        polys = self.polygon.geoms if isinstance(self.polygon, sg.MultiPolygon) else [self.polygon]
        for p in polys:
            plt.plot(*p.boundary.xy, 'b--', marker='.', label='Polygon')
            ax.add_patch(PolygonPatch(p, fc = sns.xkcd_rgb['green'], ec = sns.xkcd_rgb['green'], alpha = 0.5, zorder=1))

        for l in lines:
            plt.plot(*self.Tf_w2p(l)[:2], 'r', marker='o', label='Line')
        plt.legend()

        return fig

    @classmethod
    def create_list_from_ann3D(cls, ann_3D, render_dir, debug_plot = False):
        # Find door mask
        line_is_door_mask = np.zeros(ann_3D['planeLineMatrix'].shape[1], dtype=np.bool)
        line_is_window_mask = np.zeros_like(line_is_door_mask)
        for semantic in ann_3D['semantics']:
            if semantic['type'] == 'door':
                for id in semantic['planeID']:
                    line_is_door_mask |= ann_3D['planeLineMatrix'][id]
            elif semantic['type'] == 'window':
                for id in semantic['planeID']:
                    line_is_window_mask |= ann_3D['planeLineMatrix'][id]

        # Create all planes, assume doors closed.
        plane_list = []
        planes_with_doors_idx = []
        for idx, (p, line_mask) in enumerate(zip(ann_3D['planes'], ann_3D['planeLineMatrix'])):
            this_door_mask = line_is_door_mask[line_mask]
            sub_line_junction_matrix = ann_3D['lineJunctionMatrix'][line_mask]

            if np.any(sub_line_junction_matrix, axis=0).sum() > 2:
                p_inst = Plane(p,
                               sub_line_junction_matrix,
                               this_door_mask,
                               line_is_window_mask[line_mask],
                               ann_3D['junctionCoords'],
                               open_doors = False)
                plane_list.append(p_inst)
                if p_inst.hole_polygons:
                    planes_with_doors_idx.append(idx)


        for i in range(2):
            #Run twice since each door is two planes and therefore may remain closed unless viewed from two positions.
            door_stats = get_door_stats(plane_list, planes_with_doors_idx, render_dir, debug_plot = debug_plot)

            for dstat, p_idx in zip(door_stats, planes_with_doors_idx):

                door_is_open = [h_idx for h_idx, hstat in enumerate(dstat)
                                if hstat['total'] > 0 and hstat['door']/hstat['total'] < 0.3]
                plane_list[p_idx]._set_door_open(door_is_open)

        return plane_list

def get_door_stats(plane_list, planes_with_doors_idx, render_dir, debug_plot = False):
    # Go through the images to see which doors are closed
    door_stats =[]
    for p_idx in planes_with_doors_idx:
        door_stats.append([{'door': 0, 'total':0}]*len(plane_list[p_idx].hole_polygons))

    for room_id in os.listdir(render_dir):
        room_dir = osp.join(render_dir, room_id, 'perspective', 'full')
        for pos_id in os.listdir(room_dir):
            pos_dir = osp.join(room_dir, pos_id)
            img = cv2.imread(osp.join(pos_dir,'semantic.png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with open(osp.join(pos_dir, 'camera_pose.txt')) as f:
                pose = next(csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC))

            R, t, K = parse_camera_info(pose, img.shape[1], img.shape[0])
            T = np.block([
                [R, t],
                [np.array([0,0,0,1])]
            ])
            T_inv = np.linalg.inv(T)
            plane_list_t = [p.transform(T_inv) for p in plane_list]

            # Check for which image all of the sample points are in front of camera and not occluded by any plane
            for p_stat, plane_wd_idx in zip(door_stats, planes_with_doors_idx):
                plane_wd = plane_list_t[plane_wd_idx]
                sample_points_list = plane_wd._gen_door_sample_points()

                for h_idx, (d_stat, sample_points_w) in enumerate(zip(p_stat, sample_points_list)):

                    in_front = sample_points_w[2] > 0
                    if ~np.any(in_front):
                        # Skip if all behind camera
                        continue
                    sample_points_w = sample_points_w[:,in_front]

                    sample_points_pix_h = K@sample_points_w
                    sample_points_pix_h /= sample_points_pix_h[2]

                    inside = ((0 < sample_points_pix_h[0]) & (sample_points_pix_h[0] < img.shape[1]) &
                              (0 < sample_points_pix_h[1]) & (sample_points_pix_h[1] < img.shape[0]))

                    if ~np.any(inside):
                        # Skip if all outside image
                        continue

                    sample_points_w = sample_points_w[:,inside]
                    sample_points_pix_h = sample_points_pix_h[:,inside]

                    # Check occlusion
                    occluded_mask = np.zeros(sample_points_w.shape[1], dtype=np.bool)

                    for p_idx, plane in enumerate(plane_list_t):
                        if p_idx == plane_wd_idx:
                            continue

                        dist_frac = - plane.params[3]/(plane.params[:3]@sample_points_w + DIV_EPS)
                        possibly_occluded = np.any((0 < dist_frac) & (dist_frac < 1-CMP_EPS))

                        if not possibly_occluded:
                            continue

                        # Projection to plane along viewing ray.
                        sample_points_w_proj = dist_frac*sample_points_w

                        # Polygon check
                        occluded_mask |= plane.contains(sample_points_w_proj)

                        if np.all(occluded_mask):
                            break

                    # Door visible, sample from image
                    if not np.all(occluded_mask):
                        sample_points_pix = sample_points_pix_h[:2,~occluded_mask].astype(np.int)
                        sem_rgb = img[sample_points_pix[1],
                                     sample_points_pix[0]]
                        door_mask = np.all((sem_rgb==door_rgb) | (sem_rgb==curtain_rgb), axis=1)
                        d_stat['door'] += np.sum(door_mask)
                        d_stat['total'] += sem_rgb.shape[0]

                        # p_holes = plane_wd.get_polygon_holes()
                        # p_hole_pix_h = K@p_holes[h_idx]
                        # p_hole_pix_h /= p_hole_pix_h[2]
                        # p_hole_pix = np.clip(p_hole_pix_h[:2],
                        #                      np.array([0,0]).reshape([2,1]),
                        #                      np.array([img.shape[1], img.shape[0]]).reshape([2,1]))

                        if debug_plot:
                            plt.figure()
                            plt.imshow(img)
                            plt.plot(*sample_points_pix[:,door_mask], 'go')
                            plt.plot(*sample_points_pix[:,~door_mask], 'ro')
                            # plt.plot(*p_hole_pix, 'g.-')
                            plt.title('Door {}, Total {}, outside {}'.format(np.sum(door_mask), sem_rgb.shape[0], np.sum(~inside)))
                            plt.savefig('/host_home/plots/hawp/debug/is_door_{}_{}_{}_{}.svg'.format(plane_wd_idx, h_idx, room_id, pos_id))
                            plt.close()


    return door_stats
