// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const TORCH_VERSION: &str = "1.11.0";

#[cfg(feature = "curl")]
fn download<P: AsRef<Path>>(source_url: &str, target_file: P) -> anyhow::Result<()> {
    use curl::easy::Easy;
    use std::io::Write;

    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let mut easy = Easy::new();
    easy.url(source_url)?;
    easy.write_function(move |data| Ok(writer.write(data).unwrap()))?;
    easy.perform()?;
    let response_code = easy.response_code()?;
    if response_code == 200 {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Unexpected response code {} for {}", response_code, source_url))
    }
}

#[cfg(not(feature = "curl"))]
fn download<P: AsRef<Path>>(_source_url: &str, _target_file: P) -> anyhow::Result<()> {
    anyhow::bail!("cannot use download without the curl feature")
}

fn extract<P: AsRef<Path>>(filename: P, outpath: P) -> anyhow::Result<()> {
    let file = fs::File::open(&filename)?;
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        #[allow(deprecated)]
        let outpath = outpath.as_ref().join(file.sanitized_name());
        if !(&*file.name()).ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var(name)
}

fn check_system_location() -> Option<PathBuf> {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    match os.as_str() {
        "linux" => Path::new("/usr/lib/libtorch.so").exists().then(|| PathBuf::from("/usr")),
        _ => None,
    }
}

fn find_python() -> String {
    env::var("PYTHON3").ok().unwrap_or_else(|| {
        let candidates = if cfg!(windows) {
            ["python3.exe", "python.exe"]
        } else {
            ["python3", "python"]
        };
        for &name in &candidates {
            if Command::new(name)
                .arg("--version")
                .output()
                .ok()
                .map_or(false, |out| out.status.success())
            {
                return name.to_owned();
            }
        }
        panic!(
            "Can't find python (tried {})! Try fixing PATH or setting the PYTHON_INCLUDE_DIRS env var explicitly",
            candidates.join(", ")
        )
    })
}

fn find_python_include_dir() -> PathBuf {
    if let Ok(python_dir) = env_var_rerun("PYTHON_INCLUDE_DIRS") {
        PathBuf::from(python_dir)
    } else {
        let python = find_python();
        let output = Command::new(python)
            .arg("-c")
            .arg("from sysconfig import get_paths as gp; print(gp()['include'])")
            .output()
            .expect("Failed to run python")
            .stdout;
        let python_dir =
            String::from_utf8(output).expect("Python output not utf8").trim().to_owned();
        PathBuf::from(python_dir)
    }
}

fn prepare_libtorch_dir() -> PathBuf {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    let device = match env_var_rerun("TORCH_CUDA_VERSION") {
        Ok(cuda_env) => match os.as_str() {
            "linux" | "windows" => {
                cuda_env.trim().to_lowercase().trim_start_matches("cu").split('.').take(2).fold(
                    "cu".to_owned(),
                    |mut acc, curr| {
                        acc += curr;
                        acc
                    },
                )
            }
            os_str => panic!(
                "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                 binaries with CUDA are only available for Linux and Windows, not: {}.",
                os_str
            ),
        },
        Err(_) => "cpu".to_owned(),
    };

    if let Ok(libtorch) = env_var_rerun("LIBTORCH") {
        PathBuf::from(libtorch)
    } else if let Some(pathbuf) = check_system_location() {
        pathbuf
    } else {
        let libtorch_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("libtorch");
        if !libtorch_dir.exists() {
            fs::create_dir(&libtorch_dir).unwrap_or_default();
            let libtorch_url = match os.as_str() {
                "linux" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu102" => "%2Bcu102",
                        "cu113" => "%2Bcu113",
                        _ => ""
                    }
                ),
                "macos" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                    TORCH_VERSION
                ),
                "windows" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu102" => "%2Bcu102",
                        "cu113" => "%2Bcu113",
                        _ => ""
                    }),
                _ => panic!("Unsupported OS"),
            };

            let filename = libtorch_dir.join(format!("v{}.zip", TORCH_VERSION));
            download(&libtorch_url, &filename).unwrap();
            extract(&filename, &libtorch_dir).unwrap();
        }

        libtorch_dir.join("libtorch")
    }
}

fn make<P: AsRef<Path>>(libtorch: P, use_cuda: bool, use_hip: bool, use_python: bool) {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    let python_includes = if use_python {
        let python_include_dir = find_python_include_dir();
        vec![python_include_dir]
    } else {
        vec![]
    };
    let use_python_flag = if use_python { "1".to_owned() } else { "0".to_owned() };

    let cuda_dependency = if use_cuda || use_hip {
        "libtch/dummy_cuda_dependency.cpp"
    } else {
        "libtch/fake_cuda_dependency.cpp"
    };
    println!("cargo:rerun-if-changed=libtch/torch_api.cpp");
    println!("cargo:rerun-if-changed=libtch/torch_api.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.cpp.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_write.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_resize.h");
    println!("cargo:rerun-if-changed=libtch/stb_image.h");
    match os.as_str() {
        "linux" | "macos" => {
            let libtorch_cxx11_abi =
                env_var_rerun("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_owned());
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.as_ref().join("include"))
                .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
                .includes(python_includes)
                .flag(&format!("-Wl,-rpath={}", libtorch.as_ref().join("lib").display()))
                .flag("-std=c++14")
                .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", libtorch_cxx11_abi))
                .flag(&format!("-DWITH_PYTHON={}", use_python_flag))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        "windows" => {
            // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
            //       Not yet supported by cc=rs.
            //       https://github.com/alexcrichton/cc-rs/issues/323
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.as_ref().join("include"))
                .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
                .includes(python_includes)
                .flag(&format!("-DWITH_PYTHON={}", use_python_flag))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        _ => panic!("Unsupported OS"),
    };
}

fn main() {
    if !cfg!(feature = "doc-only") {
        let libtorch = prepare_libtorch_dir();
        // use_cuda is a hacky way to detect whether cuda is available and
        // if it's the case link to it by explicitly depending on a symbol
        // from the torch_cuda library.
        // It would be better to use -Wl,--no-as-needed but there is no way
        // to specify arbitrary linker flags at the moment.
        //
        // Once https://github.com/rust-lang/cargo/pull/8441 is available
        // we should switch to using rustc-link-arg instead e.g. with the
        // following flags:
        //   -Wl,--no-as-needed -Wl,--copy-dt-needed-entries -ltorch
        //
        // This will be available starting from cargo 1.50 but will be a nightly
        // only option to start with.
        // https://github.com/rust-lang/cargo/blob/master/CHANGELOG.md
        let use_cuda = libtorch.join("lib").join("libtorch_cuda.so").exists()
            || libtorch.join("lib").join("torch_cuda.dll").exists();
        let use_cuda_cu = libtorch.join("lib").join("libtorch_cuda_cu.so").exists()
            || libtorch.join("lib").join("torch_cuda_cu.dll").exists();
        let use_cuda_cpp = libtorch.join("lib").join("libtorch_cuda_cpp.so").exists()
            || libtorch.join("lib").join("torch_cuda_cpp.dll").exists();
        let use_hip = libtorch.join("lib").join("libtorch_hip.so").exists()
            || libtorch.join("lib").join("torch_hip.dll").exists();

        let use_python = cfg!(feature = "python");
        if use_python
            && !libtorch.join("lib").join("libtorch_python.so").exists()
            && !libtorch.join("lib").join("torch_python.dll").exists()
        {
            panic!("libtorch_python.so or torch_python.dll not found");
        }

        println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());

        make(&libtorch, use_cuda, use_hip, use_python);

        let link_type = if use_python { "dylib" } else { "static" };

        println!("cargo:rustc-link-lib=static=tch");
        if use_cuda {
            println!("cargo:rustc-link-lib={}=torch_cuda", link_type);
        }
        if use_cuda_cu {
            println!("cargo:rustc-link-lib={}=torch_cuda_cu", link_type);
        }
        if use_cuda_cpp {
            println!("cargo:rustc-link-lib={}=torch_cuda_cpp", link_type);
        }
        if use_hip {
            println!("cargo:rustc-link-lib={}=torch_hip", link_type);
        }
        println!("cargo:rustc-link-lib={}=torch_cpu", link_type);
        println!("cargo:rustc-link-lib={}=torch", link_type);
        println!("cargo:rustc-link-lib={}=c10", link_type);
        if use_hip {
            println!("cargo:rustc-link-lib={}=c10_hip", link_type);
        }
        if use_python {
            println!("cargo:rustc-link-lib={}=torch_python", link_type);
        }

        let target = env::var("TARGET").unwrap();

        if !target.contains("msvc") && !target.contains("apple") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }
}
