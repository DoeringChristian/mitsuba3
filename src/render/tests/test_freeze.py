import os
from typing import Any, Callable, List
import pytest
import drjit as dr
import mitsuba as mi
import glob
import gc
import numpy as np
from os.path import join, realpath, dirname, basename, splitext, exists

from mitsuba.scalar_rgb.test.util import find_resource

dr.set_log_level(dr.LogLevel.Trace)
dr.set_flag(dr.JitFlag.ReuseIndices, False)


def test01_cornell_box(variants_vec_rgb):
    w = 16
    h = 16

    n = 5

    k = "light.emitter.radiance.value"

    def func(scene: mi.Scene, x) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def load_scene() -> mi.Scene:
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        scene = mi.load_dict(scene, parallel = False)
        return scene


    def run(scene: mi.Scene, n: int, func: Callable[[mi.Scene, Any], mi.TensorXf]) -> List[mi.TensorXf]:
        params = mi.traverse(scene)
        value = mi.Float(params[k].x)

        images = []
        for i in range(n):
            params[k].x = value + 10.0 * i
            params.update()

            img = func(scene, params[k].x)
            dr.eval(img)

            images.append(img)

        return images

    scene = load_scene()
    images_ref = run(scene, n, func)
    scene2 = load_scene()
    images_frozen = run(scene2, n, dr.freeze(func))

    for ref, frozen in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)


def test02_cornell_box_native(variants_vec_rgb):
    if mi.MI_ENABLE_EMBREE:
        pytest.skip("EMBREE enabled")

    w = 16
    h = 16

    n = 5

    k = "light.emitter.radiance.value"

    def func(scene: mi.Scene, x) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def run(n: int, func: Callable[[mi.Scene, Any], mi.TensorXf]) -> List[mi.TensorXf]:
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        scene = mi.load_dict(scene)

        params = mi.traverse(scene)
        value = mi.Float(params[k].x)

        images = []
        for i in range(n):
            params[k].x = value + 10.0 * i
            params.update()

            img = func(scene, params[k].x)
            dr.eval(img)

            images.append(img)

        return images

    images_ref = run(n, func)
    images_frozen = run(n, dr.freeze(func))

    for ref, frozen in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)


tutorials_dir = realpath(join(dirname(__file__), "../../../tutorials"))


@pytest.mark.parametrize(
    "integrator",
    [
        "direct",
        "prb",
        # "prb_basic",
        "direct_projective",
        "prb_projective",
    ],
)
def test02_pose_estimation(variants_vec_rgb, integrator):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)
    # dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    # dr.set_flag(dr.JitFlag.OptimizeCalls, False)
    w = 128
    h = 128

    def apply_transformation(initial_vertex_positions, opt, params):
        opt["trans"] = dr.clip(opt["trans"], -0.5, 0.5)
        opt["angle"] = dr.clip(opt["angle"], -0.5, 0.5)

        trafo = (
            mi.Transform4f()
            .translate([opt["trans"].x, opt["trans"].y, 0.0])
            .rotate([0, 1, 0], opt["angle"] * 100.0)
        )

        print("ravel:")
        params["bunny.vertex_positions"] = dr.ravel(trafo @ initial_vertex_positions)

    def mse(image, image_ref):
        return dr.sum(dr.square(image - image_ref), axis=None)

    def optimize(scene, ref, initial_vertex_positions, other):
        params = mi.traverse(scene)
        print(f"{dr.grad_enabled(params['bunny.vertex_positions'])=}")

        image = mi.render(scene, params, spp=1, seed=1, seed_grad=2)

        # Evaluate the objective function from the current rendered image
        loss = mse(image, ref)
        print(f"{type(loss)=}")

        # Backpropagate through the rendering process
        dr.backward(loss)

        return image, loss

    def load_scene():
        from mitsuba.scalar_rgb import Transform4f as T

        scene = mi.cornell_box()
        del scene["large-box"]
        del scene["small-box"]
        del scene["green-wall"]
        del scene["red-wall"]
        del scene["floor"]
        del scene["ceiling"]
        scene["bunny"] = {
            "type": "ply",
            "filename": f"{tutorials_dir}/scenes/meshes/bunny.ply",
            "to_world": T().scale(6.5),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": (0.3, 0.3, 0.75)},
            },
        }
        scene["integrator"] = {
            "type": "prb",
        }
        scene["sensor"]["film"] = {
            "type": "hdrfilm",
            "width": w,
            "height": h,
            "rfilter": {"type": "gaussian"},
            "sample_border": True,
        }

        scene = mi.load_dict(scene, parallel=False)
        return scene

    def run(scene: mi.Scene, optimize, n) -> tuple[mi.TensorXf, mi.Point3f, mi.Float]:
        params = mi.traverse(scene)

        params.keep("bunny.vertex_positions")
        initial_vertex_positions = dr.unravel(
            mi.Point3f, params["bunny.vertex_positions"]
        )

        image_ref = mi.render(scene, spp=4)

        opt = mi.ad.Adam(lr=0.025)
        opt["angle"] = mi.Float(0.25)
        opt["trans"] = mi.Point3f(0.1, -0.25, 0.0)

        for i in range(n):
            params = mi.traverse(scene)
            params.keep("bunny.vertex_positions")

            apply_transformation(initial_vertex_positions, opt, params)
            print(f"{dr.grad_enabled(params['bunny.vertex_positions'])=}")

            with dr.profile_range("optimize"):
                image, loss = optimize(
                    scene,
                    image_ref,
                    initial_vertex_positions,
                    [
                        params["bunny.vertex_positions"],
                    ],
                )

            opt.step()

        image_final = mi.render(scene, spp=4, seed=1, seed_grad=2)

        return image_final, opt["trans"], opt["angle"]

    n = 10

    # NOTE:
    # In this cas, we have to use the same scene object
    # for the frozen and non-frozen case, as re-loading
    # the scene causes mitsuba to render different images,
    # leading to diverging descent traijectories.

    scene = load_scene()
    params = mi.traverse(scene)
    initial_vertex_positions = mi.Float(params["bunny.vertex_positions"])

    print("Reference:")
    img_ref, trans_ref, angle_ref = run(scene, optimize, n)

    # Reset parameters:
    params["bunny.vertex_positions"] = initial_vertex_positions
    params.update()

    print("Frozen:")
    img_frozen, trans_frozen, angle_frozen = run(scene, optimize, n)

    # NOTE: cannot compare results as errors accumulate and the result will never be the same.

    assert dr.allclose(trans_ref, trans_frozen)
    assert dr.allclose(angle_ref, angle_frozen)
    if integrator != "prb_projective":
        print(f"{dr.max(dr.abs(img_ref - img_frozen), axis=None)=}")
        assert dr.allclose(img_ref, img_frozen, atol = 1e-4)


def test03_optimize_color(variants_vec_rgb):
    k = "red.reflectance.value"
    w = 128
    h = 128
    n = 10

    def mse(image, image_ref):
        return dr.sum(dr.square(image - image_ref), axis=None)

    def optimize(scene, image_ref):
        params = mi.traverse(scene)
        params.keep(k)

        image = mi.render(scene, params, spp=1)

        loss = mse(image, image_ref)

        dr.backward(loss)

        return image, loss

    def run(n: int, optimize):
        scene = mi.cornell_box()
        scene["integrator"] = {
            "type": "prb",
        }
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        scene = mi.load_dict(scene)

        image_ref = mi.render(scene, spp=512)

        params = mi.traverse(scene)
        params.keep(k)

        opt = mi.ad.Adam(lr=0.05)
        opt[k] = mi.Color3f(0.01, 0.2, 0.9)

        for i in range(n):
            params = mi.traverse(scene)
            params.keep(k)
            params.update(opt)

            image, loss = optimize(scene, image_ref)

            opt.step()

        return image, opt[k]

    image_ref, param_ref = run(n, optimize)

    image_frozen, param_frozen = run(n, dr.freeze(optimize))

    # Optimizing the reflectance is not as prone to divergence,
    # therefore we can test if the two methods produce the same results
    assert dr.allclose(param_ref, param_frozen)


@pytest.mark.parametrize(
    "bsdf",
    [
        "diffuse",
        "dielectric",
        "thindielectric",
        "roughdielectric",
        "conductor",
        "roughconductor",
        "hair",
        "plastic",
        "roughplastic",
        "bumpmap",
        "normalmap",
        "blendbsdf",
        "mask",
        "twosided",
        "principled",
        "principledthin",
        "custom",
    ],
)
def test04_bsdf(variants_vec_rgb, bsdf):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)

    if bsdf == "custom":

        class MyBSDF(mi.BSDF):
            def __init__(self, props):
                mi.BSDF.__init__(self, props)

                # Read 'eta' and 'tint' properties from `props`
                self.eta = 1.33
                if props.has_property("eta"):
                    self.eta = props["eta"]

                self.tint = props["tint"]

                # Set the BSDF flags
                reflection_flags = (
                    mi.BSDFFlags.DeltaReflection
                    | mi.BSDFFlags.FrontSide
                    | mi.BSDFFlags.BackSide
                )
                transmission_flags = (
                    mi.BSDFFlags.DeltaTransmission
                    | mi.BSDFFlags.FrontSide
                    | mi.BSDFFlags.BackSide
                )
                self.m_components = [reflection_flags, transmission_flags]
                self.m_flags = reflection_flags | transmission_flags

            def sample(self, ctx, si, sample1, sample2, active):
                # Compute Fresnel terms
                cos_theta_i = mi.Frame3f.cos_theta(si.wi)
                r_i, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, self.eta)
                t_i = dr.maximum(1.0 - r_i, 0.0)

                # Pick between reflection and transmission
                selected_r = (sample1 <= r_i) & active

                # Fill up the BSDFSample struct
                bs = mi.BSDFSample3f()
                bs.pdf = dr.select(selected_r, r_i, t_i)
                bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
                bs.sampled_type = dr.select(
                    selected_r,
                    mi.UInt32(+mi.BSDFFlags.DeltaReflection),
                    mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
                )
                bs.wo = dr.select(
                    selected_r,
                    mi.reflect(si.wi),
                    mi.refract(si.wi, cos_theta_t, eta_ti),
                )
                bs.eta = dr.select(selected_r, 1.0, eta_it)

                # For reflection, tint based on the incident angle (more tint at grazing angle)
                value_r = dr.lerp(
                    mi.Color3f(self.tint),
                    mi.Color3f(1.0),
                    dr.clip(cos_theta_i, 0.0, 1.0),
                )

                # For transmission, radiance must be scaled to account for the solid angle compression
                value_t = mi.Color3f(1.0) * dr.square(eta_ti)

                value = dr.select(selected_r, value_r, value_t)

                return (bs, value)

            def eval(self, ctx, si, wo, active):
                return 0.0

            def pdf(self, ctx, si, wo, active):
                return 0.0

            def eval_pdf(self, ctx, si, wo, active):
                return 0.0, 0.0

            def traverse(self, callback):
                callback.put_parameter("tint", self.tint, mi.ParamFlags.Differentiable)

            def parameters_changed(self, keys):
                print("🏝️ there is nothing to do here 🏝️")

            def to_string(self):
                return "MyBSDF[\n" "    eta=%s,\n" "    tint=%s,\n" "]" % (
                    self.eta,
                    self.tint,
                )

        mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))

    w = 16
    h = 16

    n = 5

    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def run(
        scene: mi.Scene, n: int, func: Callable[[mi.Scene], mi.TensorXf]
    ) -> List[mi.TensorXf]:
        images = []
        for i in range(n):
            img = func(scene)
            dr.eval(img)

            images.append(img)

        return images

    def load_scene(bsdf: str):
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        if bsdf == "twosided":
            scene["white"] = {
                "type": "twosided",
                "material": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": 0.4},
                },
            }
        elif bsdf == "mask":
            scene["white"] = {
                "type": "mask",
                # Base material: a two-sided textured diffuse BSDF
                "material": {
                    "type": "twosided",
                    "bsdf": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "bitmap",
                            "filename": find_resource(
                                "resources/data/common/textures/wood.jpg"
                            ),
                        },
                    },
                },
                # Fetch the opacity mask from a monochromatic texture
                "opacity": {
                    "type": "bitmap",
                    "filename": find_resource(
                        "resources/data/common/textures/leaf_mask.png"
                    ),
                },
            }
        elif bsdf == "bumpmap":
            scene["white"] = {
                "type": "bumpmap",
                "arbitrary": {
                    "type": "bitmap",
                    "raw": True,
                    "filename": find_resource(
                        "resources/data/common/textures/floor_tiles_bumpmap.png"
                    ),
                },
                "bsdf": {"type": "roughplastic"},
            }
        elif bsdf == "normalmap":
            scene["white"] = {
                "type": "normalmap",
                "normalmap": {
                    "type": "bitmap",
                    "raw": True,
                    "filename": find_resource(
                        "resources/data/common/textures/floor_tiles_normalmap.jpg"
                    ),
                },
                "bsdf": {"type": "roughplastic"},
            }
        elif bsdf == "blendbsdf":
            scene["white"] = {
                "type": "blendbsdf",
                "weight": {
                    "type": "bitmap",
                    "filename": find_resource(
                        "resources/data/common/textures/noise_01.jpg"
                    ),
                },
                "bsdf_0": {"type": "conductor"},
                "bsdf_1": {"type": "roughplastic", "diffuse_reflectance": 0.1},
            }
        elif bsdf == "diffuse":
            scene["white"] = {
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "filename": find_resource(
                        "resources/data/common/textures/wood.jpg"
                    ),
                },
            }
        elif bsdf == "custom":
            scene["white"] = {
                "type": "mybsdf",
                "tint": [0.2, 0.9, 0.2],
                "eta": 1.33,
            }
        else:
            scene["white"] = {
                "type": bsdf,
            }
        scene = mi.load_dict(scene)
        return scene

    scene = load_scene(bsdf)
    images_ref = run(scene, n, func)
    scene = load_scene(bsdf)
    images_frozen = run(scene, n, dr.freeze(func))

    for ref, frozen in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)


@pytest.mark.parametrize(
    "emitter",
    [
        "area",
        "point",
        "constant",
        "envmap",
        "spot",
        "projector",
        "directional",
        "directionalarea",
    ],
)
def test05_emitter(variants_vec_rgb, emitter):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)

    w = 16
    h = 16

    n = 5

    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def load_scene(emitter):
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h

        if emitter == "point":
            scene["emitter"] = {
                "type": "point",
                "position": [0.0, 0.0, 0.0],
                "intensity": {
                    "type": "rgb",
                    "value": mi.ScalarColor3d(50, 50, 50),
                },
            }
        elif emitter == "constant":
            scene["emitter"] = {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": 1.0,
                },
            }
        elif emitter == "envmap":
            scene["emitter"] = {
                "type": "envmap",
                "filename": find_resource(
                    "resources/data/scenes/matpreview/envmap.exr"
                ),
            }
        elif emitter == "spot":
            scene["emitter"] = {
                "type": "spot",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[0, 0, 0],
                    target=[0, -1, 0],
                    up=[0, 0, 1],
                ),
                "intensity": {
                    "type": "rgb",
                    "value": 1.0,
                },
            }
        elif emitter == "projector":
            scene["emitter"] = {
                "type": "projector",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[0, 0, 0],
                    target=[0, -1, 0],
                    up=[0, 0, 1],
                ),
                "fov": 45,
                "irradiance": {
                    "type": "bitmap",
                    "filename": find_resource(
                        "resources/data/common/textures/flower.bmp"
                    ),
                },
            }
        elif emitter == "directional":
            scene["emitter"] = {
                "type": "directional",
                "direction": [0, 0, -1],
                "irradiance": {
                    "type": "rgb",
                    "value": 1.0,
                },
            }
        elif emitter == "directionalarea":
            scene["light"]["emitter"] = {
                "type": "directionalarea",
                "radiance": {
                    "type": "rgb",
                    "value": 1.0,
                },
            }

        scene = mi.load_dict(scene, parallel=False)
        return scene

    def run(n: int, func: Callable[[mi.Scene], mi.TensorXf]) -> List[mi.TensorXf]:
        scene = load_scene(emitter)

        images = []
        for i in range(n):
            img = func(scene)
            dr.eval(img)

            images.append(img)

        return images

    images_ref = run(n, func)
    print("scene1")
    # scene = load_scene(emitter)
    # del scene
    # gc.collect()
    # gc.collect()
    print("scene2")
    images_frozen = run(n, dr.freeze(func))

    for ref, frozen in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)


@pytest.mark.parametrize(
    "integrator",
    [
        "direct",
        "path",
        "prb",
        "prb_basic",
        "direct_projective",
        "prb_projective",
        "moment",
        "ptracer",
        "depth",
    ],
)
def test06_integrators(variants_vec_rgb, integrator):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)

    w = 16
    h = 16

    n = 5

    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def load_scene():
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h

        if integrator == "path":
            scene["integrator"] = {
                "type": "path",
                "max_depth": 4,
            }
        elif integrator == "direct":
            scene["integrator"] = {
                "type": "direct",
            }
        elif integrator == "prb":
            scene["integrator"] = {
                "type": "prb",
            }
        elif integrator == "prb_basic":
            scene["integrator"] = {
                "type": "prb_basic",
            }
        elif integrator == "direct_projective":
            scene["integrator"] = {
                "type": "direct_projective",
            }
        elif integrator == "prb_projective":
            scene["integrator"] = {
                "type": "prb_projective",
            }
        elif integrator == "moment":
            scene["integrator"] = {
                "type": "moment",
                "nested": {
                    "type": "path",
                },
            }
        elif integrator == "ptracer":
            scene["integrator"] = {
                "type": "ptracer",
                "max_depth": 8,
            }
        elif integrator == "depth":
            scene["integrator"] = {"type": "depth"}

        scene = mi.load_dict(scene, parallel=False)
        return scene

    def run(n: int, func: Callable[[mi.Scene], mi.TensorXf]) -> List[mi.TensorXf]:
        scene = load_scene()

        images = []
        for i in range(n):
            img = func(scene)
            dr.eval(img)

            images.append(img)

        return images

    # scene = load_scene()
    images_ref = run(n, func)
    images_frozen = run(n, dr.freeze(func))

    for ref, frozen in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)


@pytest.mark.parametrize(
    "shape",
    [
        "mesh",
        "disk",
        "cylinder",
        "bsplinecurve",
        "linearcurve",
        "sdfgrid",
        # "instance",
        "sphere",
    ],
)
def test07_shape(variants_vec_rgb, shape):
    w = 128
    h = 128

    n = 5
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)
    # dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    # dr.set_flag(dr.JitFlag.OptimizeCalls, False)

    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def load_scene():
        from mitsuba.scalar_rgb import Transform4f as T

        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        del scene["small-box"]
        del scene["large-box"]

        if shape == "mesh":
            scene["shape"] = {
                "type": "ply",
                "filename": find_resource("resources/data/common/meshes/teapot.ply"),
                "to_world": T().scale(0.1),
            }
        elif shape == "disk":
            scene["shape"] = {
                "type": "disk",
                "material": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "checkerboard",
                        "to_uv": mi.ScalarTransform4f().rotate(
                            axis=[1, 0, 0], angle=45
                        ),
                    },
                },
            }
        elif shape == "cylinder":
            scene["shape"] = {
                "type": "cylinder",
                "radius": 0.3,
                "material": {"type": "diffuse"},
                "to_world": mi.ScalarTransform4f().rotate(axis=[1, 0, 0], angle=10),
            }
        elif shape == "bsplinecurve":
            scene["shape"] = {
                "type": "bsplinecurve",
                "to_world": mi.ScalarTransform4f()
                .scale(1.0)
                .rotate(axis=[0, 1, 0], angle=45),
                "filename": find_resource("resources/data/common/meshes/curve.txt"),
                "silhouette_sampling_weight": 1.0,
            }
        elif shape == "linearcurve":
            scene["shape"] = {
                "type": "linearcurve",
                "to_world": mi.ScalarTransform4f()
                .translate([0, -1, 0])
                .scale(1)
                .rotate(axis=[0, 1, 0], angle=45),
                "filename": find_resource("resources/data/common/meshes/curve.txt"),
            }
        elif shape == "sdfgrid":
            scene["shape"] = {
                "type": "sdfgrid",
                "bsdf": {"type": "diffuse"},
                "filename": find_resource(
                    "resources/data/docs/scenes/sdfgrid/torus_sdfgrid.vol"
                ),
            }
        elif shape == "instance":
            scene["sg"] = {
                "type": "shapegroup",
                "first_object": {
                    "type": "ply",
                    "filename": find_resource(
                        "resources/data/common/meshes/teapot.ply"
                    ),
                    "bsdf": {
                        "type": "roughconductor",
                    },
                    "to_world": mi.ScalarTransform4f()
                    .translate([0.5, 0, 0])
                    .scale([0.2, 0.2, 0.2]),
                },
                "second_object": {
                    "type": "sphere",
                    "to_world": mi.ScalarTransform4f()
                    .translate([-0.5, 0, 0])
                    .scale([0.2, 0.2, 0.2]),
                    "bsdf": {
                        "type": "diffuse",
                    },
                },
            }
            scene["first_instance"] = {
                "type": "instance",
                "shapegroup": {"type": "ref", "id": "sg"},
            }
        elif shape == "sphere":
            scene["shape"] = {
                "type": "sphere",
                "center": [0, 0, 0],
                "radius": 0.5,
                "bsdf": {"type": "diffuse"},
            }

        scene = mi.load_dict(scene, parallel=False)
        return scene

    def run(
        scene, n: int, func: Callable[[mi.Scene], mi.TensorXf]
    ) -> List[mi.TensorXf]:
        images = []
        for i in range(n):
            img = func(scene)
            dr.eval(img)

            images.append(img)

        return images

    scene = load_scene()
    images_ref = run(scene, n, func)
    images_frozen = run(scene, n, dr.freeze(func))

    for (i, (ref, frozen)) in enumerate(zip(images_ref, images_frozen)):
        os.makedirs(f"out/{shape}", exist_ok=True)
        mi.util.write_bitmap(f"out/{shape}/ref{i}.jpg", ref)
        mi.util.write_bitmap(f"out/{shape}/frozen{i}.jpg", frozen)

    for i, (ref, frozen) in enumerate(zip(images_ref, images_frozen)):
        assert dr.allclose(ref, frozen)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test07_optimizer(variants_vec_rgb, optimizer):
    k = "red.reflectance.value"
    w = 128
    h = 128
    n = 10

    def mse(image, image_ref):
        return dr.sum(dr.square(image - image_ref), axis=None)

    def optimize(scene, opt, image_ref):
        params = mi.traverse(scene)
        params.update(opt)

        image = mi.render(scene, params, spp=1)

        loss = mse(image, image_ref)

        dr.backward(loss)

        opt.step()

        return image, loss

    def run(n: int, optimize):
        scene = mi.cornell_box()
        scene["integrator"] = {
            "type": "prb",
        }
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        scene = mi.load_dict(scene)

        image_ref = mi.render(scene, spp=512)

        params = mi.traverse(scene)
        params.keep(k)

        if optimizer == "adam":
            opt = mi.ad.Adam(lr=0.05)
        elif optimizer == "sgd":
            opt = mi.ad.SGD(lr=0.005)
        opt[k] = mi.Color3f(0.01, 0.2, 0.9)

        for i in range(n):
            image, loss = optimize(scene, opt, image_ref)

        return image, opt[k]

    image_ref, param_ref = run(n, optimize)

    frozen = dr.freeze(optimize)
    image_frozen, param_frozen = run(n, frozen)
    assert frozen.n_recordings == 2

    # Optimizing the reflectance is not as prone to divergence,
    # therefore we can test if the two methods produce the same results
    assert dr.allclose(param_ref, param_frozen)
    
@pytest.mark.parametrize(
    "medium",
    [
        "homogeneous",
        "heterogeneous",
    ],
)
def test08_medium(variants_vec_rgb, medium):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)

    w = 16
    h = 16

    n = 5

    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def load_scene():
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        scene["integrator"] = {"type": "volpath"}
        scene["sensor"]["medium"] = {"type": "ref", "id": "fog"}

        if medium == "homogeneous":
            scene["sensor"]["medium"] = {
                "type": "homogeneous",
                "albedo": {"type": "rgb", "value": [0.99, 0.9, 0.96]},
                "sigma_t": {
                    "type": "rgb",
                    "value": [0.5, 0.25, 0.8],
                },
            }
        elif medium == "heterogeneous":
            scene["sensor"]["medium"] = {
                "type": "heterogeneous",
                "albedo": {"type": "rgb", "value": [0.99, 0.9, 0.96]},
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": find_resource("resources/data/docs/scenes/textures/albedo.vol"),
                },
            }

        scene = mi.load_dict(scene, parallel=False)
        return scene

    def run(n: int, func: Callable[[mi.Scene], mi.TensorXf]) -> List[mi.TensorXf]:
        scene = load_scene()

        images = []
        for i in range(n):
            img = func(scene)
            dr.eval(img)

            images.append(img)

        return images

    # scene = load_scene()
    images_ref = run(n, func)
    frozen = dr.freeze(func)
    images_frozen = run(n, frozen)
    
    assert frozen.n_recordings < n
    
    for (i, (ref, frozen)) in enumerate(zip(images_ref, images_frozen)):
        os.makedirs(f"out/{medium}", exist_ok=True)
        mi.util.write_bitmap(f"out/{medium}/ref{i}.jpg", ref)
        mi.util.write_bitmap(f"out/{medium}/frozen{i}.jpg", frozen)

    for ref, frozen in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)


@pytest.mark.parametrize(
    "sampler",
    [
        "independent",
        "stratified",
        "multijitter",
        "orthogonal",
        "ldsampler",
    ],
)
def test09_sampler(variants_vec_rgb, sampler):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)

    w = 16
    h = 16

    n = 5

    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def load_scene():
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h

        scene["sensor"]["sampler"] = {"type": sampler}

        scene = mi.load_dict(scene, parallel=False)
        return scene

    def run(n: int, func: Callable[[mi.Scene], mi.TensorXf]) -> List[mi.TensorXf]:
        scene = load_scene()

        images = []
        for i in range(n):
            img = func(scene)
            dr.eval(img)

            images.append(img)

        return images

    # scene = load_scene()
    images_ref = run(n, func)
    frozen = dr.freeze(func)
    images_frozen = run(n, frozen)

    assert frozen.n_recordings < n
    
    for (i, (ref, frozen)) in enumerate(zip(images_ref, images_frozen)):
        os.makedirs(f"out/{sampler}", exist_ok=True)
        mi.util.write_bitmap(f"out/{sampler}/ref{i}.jpg", ref)
        mi.util.write_bitmap(f"out/{sampler}/frozen{i}.jpg", frozen)

    for ref, frozen in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)
        