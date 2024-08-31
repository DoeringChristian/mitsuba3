from typing import Any, Callable, List
import pytest
import drjit as dr
import mitsuba as mi
import glob
import gc
from os.path import join, realpath, dirname, basename, splitext, exists

from mitsuba.scalar_rgb.test.util import find_resource

def test01_cornell_box(variants_vec_rgb):
    
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

    for (ref, frozen) in zip(images_ref, images_frozen):
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

    for (ref, frozen) in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)
        
        
tutorials_dir = realpath(join(dirname(__file__), '../../../tutorials'))

def test02_pose_estimation(variants_vec_rgb):

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

        image = mi.render(scene, params, spp=1, seed = 1, seed_grad = 2)

        # Evaluate the objective function from the current rendered image
        loss = mse(image, ref)
        print(f"{type(loss)=}")

        # Backpropagate through the rendering process
        dr.backward(loss)

        return image, loss

    def run(optimize, n) -> tuple[mi.TensorXf, mi.Point3f, mi.Float]:
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
        
        scene = mi.load_dict(scene)
        params = mi.traverse(scene)
        
        params.keep("bunny.vertex_positions")
        initial_vertex_positions = dr.unravel(mi.Point3f, params["bunny.vertex_positions"])
        
        image_ref = mi.render(scene, spp=4)
        
        opt = mi.ad.Adam(lr=0.025)
        opt["angle"] = mi.Float(0.25)
        opt["trans"] = mi.Point3f(0.1, -0.25, 0.0)

        for i in range(n):
            params = mi.traverse(scene)
            params.keep("bunny.vertex_positions")
            
            apply_transformation(initial_vertex_positions, opt, params)
            
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

        image_final = mi.render(scene, spp=4, seed = 1, seed_grad = 2)

        return image_final, opt["trans"], opt["angle"]


    n = 10

    print("Reference:")
    img_ref, trans_ref, angle_ref = run(optimize, n)
    print("Frozen:")
    img_frozen, trans_frozen, angle_frozen = run(dr.freeze(optimize), n)

    # NOTE: cannot compare results as errors accumulate and the result will never be the same.
    
    # assert dr.allclose(trans_ref, trans_frozen)
    # assert dr.allclose(angle_ref, angle_frozen)
    # assert dr.allclose(img_ref, img_frozen)

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
    ],
)
def test04_bsdf(variants_vec_rgb, bsdf):
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

    def run(scene: mi.Scene, n: int, func: Callable[[mi.Scene], mi.TensorXf]) -> List[mi.TensorXf]:

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
                        "reflectance": {"type": "bitmap", "filename": find_resource("resources/data/common/textures/wood.jpg")},
                    },
                },
                # Fetch the opacity mask from a monochromatic texture
                "opacity": {"type": "bitmap", "filename": find_resource("resources/data/common/textures/leaf_mask.png")},
            }
        elif bsdf == "bumpmap":
            scene["white"] = {
                "type": "bumpmap",
                "arbitrary": {"type": "bitmap", "raw": True, "filename": find_resource("resources/data/common/textures/floor_tiles_bumpmap.png")},
                "bsdf": {"type": "roughplastic"},
            }
        elif bsdf == "normalmap":
            scene["white"] = {
                "type": "normalmap",
                "normalmap": {
                    "type": "bitmap",
                    "raw": True,
                    "filename": find_resource("resources/data/common/textures/floor_tiles_normalmap.jpg"),
                },
                "bsdf": {"type": "roughplastic"},
            }
        elif bsdf == "blendbsdf":
            scene["white"] = {
                "type": "blendbsdf",
                "weight": {"type": "bitmap", "filename": find_resource("resources/data/common/textures/noise_01.jpg")},
                "bsdf_0": {"type": "conductor"},
                "bsdf_1": {"type": "roughplastic", "diffuse_reflectance": 0.1},
            }
        elif bsdf == "diffuse":
            scene["white"] = {
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "filename": find_resource("resources/data/common/textures/wood.jpg"),
                }
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
    
    for (ref, frozen) in zip(images_ref, images_frozen):
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

        scene = mi.load_dict(scene)
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
    scene = load_scene(emitter)
    del scene
    gc.collect()
    gc.collect()
    images_frozen = run(n, dr.freeze(func))
    
    for (ref, frozen) in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)
        
        
@pytest.mark.parametrize(
    "integrator",
    [
        "direct",
        "path",
        "prb",
        # "prb_basic",
        "direct_projective",
        # "prb_projective",
        "moment",
        "ptracer",
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

        scene = mi.load_dict(scene)
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
    
    for (ref, frozen) in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)
