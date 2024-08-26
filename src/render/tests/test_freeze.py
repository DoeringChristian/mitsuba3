from typing import Any, Callable, List
import pytest
import drjit as dr
import mitsuba as mi
import glob
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

    def run(n: int, func: Callable[mi.Scene, Any]) -> List[mi.TensorXf]:
        
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

        image = mi.render(scene, params, spp=1, seed = 0)

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

        image_final = mi.render(scene, spp=4, seed = 0)

        return image_final, opt["trans"], opt["angle"]


    n = 10

    print("Reference:")
    img_ref, trans_ref, angle_ref = run(optimize, n)
    print("Frozen:")
    img_frozen, trans_frozen, angle_frozen = run(dr.freeze(optimize), n)

    # NOTE: cannot compare results as errors accumulate and the result will never be the same.
    
    # assert dr.allclose(trans_ref, trans_frozen, 0.1)
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
        "principled",
        "principledthin",
    ],
)
def test04_bsdf(variants_vec_rgb, bsdf):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    
    w = 16
    h = 16

    n = 5
    
    k = "light.emitter.radiance.value"
    
    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def run(scene: mi.Scene, n: int, func: Callable[mi.Scene, Any]) -> List[mi.TensorXf]:

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
        
